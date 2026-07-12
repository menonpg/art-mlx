#!/usr/bin/env python3
"""
ART-MLX: SQL Query Assistant

Train an AI to write better SQL queries by learning from
execution results and query performance.

INDUSTRY: Data Analytics / BI Tools
USE CASE: Natural language to SQL that actually works

This example shows how to use GRPO to fine-tune an LLM to:
1. Generate syntactically correct SQL
2. Return correct results (match expected output)
3. Write efficient queries (avoid slow patterns)
4. Handle edge cases (NULLs, empty tables, etc.)

Usage:
    python examples/industry/sql_assistant.py
"""

import asyncio
import random
import re
import time
from dataclasses import dataclass

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
except ImportError:
    print("MLX not available. Install with: pip install mlx mlx-lm")
    exit(1)


# ============================================================
# Sample Database Schema and Questions
# ============================================================

SCHEMA = """
-- E-commerce database schema

CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP,
    country VARCHAR(50)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    total_amount DECIMAL(10,2),
    status VARCHAR(20),  -- 'pending', 'completed', 'cancelled'
    created_at TIMESTAMP
);

CREATE TABLE order_items (
    id INT PRIMARY KEY,
    order_id INT REFERENCES orders(id),
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2)
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock_quantity INT
);
"""

SAMPLE_QUESTIONS = [
    {
        "question": "How many orders were placed last month?",
        "expected_tables": ["orders"],
        "expected_keywords": ["COUNT", "WHERE"],
        "correct_sql": "SELECT COUNT(*) FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE)",
    },
    {
        "question": "What are the top 5 customers by total spend?",
        "expected_tables": ["customers", "orders"],
        "expected_keywords": ["SUM", "GROUP BY", "ORDER BY", "LIMIT"],
        "correct_sql": "SELECT c.name, SUM(o.total_amount) as total_spend FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_spend DESC LIMIT 5",
    },
    {
        "question": "Which products are out of stock?",
        "expected_tables": ["products"],
        "expected_keywords": ["WHERE"],
        "correct_sql": "SELECT name FROM products WHERE stock_quantity = 0",
    },
    {
        "question": "What's the average order value by country?",
        "expected_tables": ["customers", "orders"],
        "expected_keywords": ["AVG", "JOIN", "GROUP BY"],
        "correct_sql": "SELECT c.country, AVG(o.total_amount) as avg_order_value FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.country",
    },
    {
        "question": "List all cancelled orders with customer names",
        "expected_tables": ["customers", "orders"],
        "expected_keywords": ["JOIN", "WHERE"],
        "correct_sql": "SELECT o.id, c.name, o.total_amount, o.created_at FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'cancelled'",
    },
    {
        "question": "What's our best selling product category?",
        "expected_tables": ["products", "order_items"],
        "expected_keywords": ["SUM", "JOIN", "GROUP BY", "ORDER BY"],
        "correct_sql": "SELECT p.category, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.category ORDER BY total_sold DESC LIMIT 1",
    },
]


# ============================================================
# SQL Validation (Simulated database execution)
# ============================================================

def extract_sql(response: str) -> str:
    """Extract SQL from model response."""
    # Try to find SQL in code blocks
    code_block = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    
    code_block = re.search(r'```\s*(SELECT.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    
    # Try to find SELECT statement directly
    select_match = re.search(r'(SELECT\s+.*?)(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return response.strip()


def score_sql(question_data: dict, generated_sql: str) -> tuple[float, list[str]]:
    """
    Score generated SQL. In production, you would:
    - Actually execute the query
    - Compare results to expected output
    - Measure query performance
    
    For training, we use heuristics.
    """
    score = 0.0
    feedback = []
    
    sql_upper = generated_sql.upper()
    
    # Check for SELECT (must have)
    if "SELECT" not in sql_upper:
        return 0.0, ["no_select"]
    score += 0.2
    feedback.append("+has_select")
    
    # Check for expected tables
    for table in question_data["expected_tables"]:
        if table.upper() in sql_upper:
            score += 0.1
            feedback.append(f"+uses_{table}")
        else:
            feedback.append(f"-missing_{table}")
    
    # Check for expected keywords
    for keyword in question_data["expected_keywords"]:
        if keyword.upper() in sql_upper:
            score += 0.1
            feedback.append(f"+has_{keyword.lower()}")
    
    # Penalize SELECT *
    if "SELECT *" in sql_upper or "SELECT  *" in sql_upper:
        score -= 0.1
        feedback.append("-select_star")
    
    # Bonus for proper formatting
    if "JOIN" in sql_upper and "ON" in sql_upper:
        score += 0.1
        feedback.append("+proper_join")
    
    # Penalize missing semicolon (style)
    if not generated_sql.strip().endswith(";"):
        score -= 0.05
        feedback.append("-no_semicolon")
    
    # Check for common anti-patterns
    if "SELECT *" in sql_upper and "WHERE 1=1" in sql_upper:
        score -= 0.2
        feedback.append("-antipattern")
    
    return min(max(score, 0), 1), feedback


# ============================================================
# SQL Agent
# ============================================================

@dataclass
class SQLAttempt:
    question: str
    sql: str
    score: float
    feedback: list[str]


class SQLAssistant:
    """Agent that converts natural language to SQL."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = make_sampler(temp=0.5)  # Lower temp for code
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL from a natural language question."""
        prompt = f"""You are a SQL expert. Given the following database schema and question, write a SQL query.

SCHEMA:
{SCHEMA}

QUESTION: {question}

Write a clean, efficient SQL query. Use proper JOINs, avoid SELECT *, and include a semicolon.

SQL:
```sql"""

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=200,
            sampler=self.sampler,
            verbose=False
        )
        
        return extract_sql(response)


# ============================================================
# GRPO Trainer
# ============================================================

class SQLTrainer:
    """Train SQL assistant using GRPO."""
    
    def __init__(self, model, tokenizer, learning_rate: float = 1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(learning_rate=learning_rate)
        self.agent = SQLAssistant(model, tokenizer)
    
    def train_step(self, questions: list[dict] = None, attempts_per_q: int = 4) -> dict:
        """Train on a batch of questions."""
        if questions is None:
            questions = random.sample(SAMPLE_QUESTIONS, min(2, len(SAMPLE_QUESTIONS)))
        
        all_attempts = []
        
        for q_data in questions:
            for _ in range(attempts_per_q):
                sql = self.agent.generate_sql(q_data["question"])
                score, feedback = score_sql(q_data, sql)
                all_attempts.append(SQLAttempt(
                    question=q_data["question"],
                    sql=sql,
                    score=score,
                    feedback=feedback,
                ))
        
        # GRPO: compute advantages
        scores = [a.score for a in all_attempts]
        mean_score = sum(scores) / len(scores)
        
        total_loss = 0.0
        num_updates = 0
        
        for attempt in all_attempts:
            advantage = attempt.score - mean_score
            
            prompt = f"Question: {attempt.question}\nSQL:"
            target = f" {attempt.sql}"
            
            full_text = prompt + target
            tokens = self.tokenizer.encode(full_text)
            if len(tokens) < 2:
                continue
            
            input_ids = mx.array([tokens[:-1]])
            labels = mx.array([tokens[1:]])
            
            def loss_fn():
                logits = self.model(input_ids)
                log_probs = nn.log_softmax(logits, axis=-1)
                target_log_probs = mx.take_along_axis(
                    log_probs[0], labels[0, :, None], axis=-1
                ).squeeze(-1)
                return -mx.mean(target_log_probs) * advantage
            
            loss, grads = nn.value_and_grad(self.model, loss_fn)()
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
            
            total_loss += float(loss)
            num_updates += 1
        
        return {
            "loss": total_loss / max(num_updates, 1),
            "mean_score": mean_score,
            "best_score": max(scores),
            "num_updates": num_updates,
        }


# ============================================================
# Main
# ============================================================

async def main():
    print("=" * 60)
    print("ART-MLX: SQL Query Assistant")
    print("=" * 60)
    print()
    
    # Load model
    model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading: {model_name}")
    
    start = time.time()
    model, tokenizer = load(model_name)
    print(f"✓ Loaded in {time.time() - start:.1f}s")
    
    # Apply LoRA
    model.freeze()
    linear_to_lora_layers(model, num_layers=8, config={"rank": 8, "scale": 20.0, "dropout": 0.0})
    num_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"✓ LoRA adapters: {num_trainable:,} trainable params")
    print()
    
    trainer = SQLTrainer(model, tokenizer)
    
    # Test before training
    test_q = SAMPLE_QUESTIONS[1]  # Top 5 customers
    print("=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    print(f"\nQuestion: {test_q['question']}")
    sql = trainer.agent.generate_sql(test_q["question"])
    score, feedback = score_sql(test_q, sql)
    print(f"\nGenerated SQL:\n{sql}")
    print(f"\nScore: {score:.2f} | {', '.join(feedback)}")
    
    # Training
    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for step in range(1, 9):
        start = time.time()
        metrics = trainer.train_step(attempts_per_q=4)
        elapsed = time.time() - start
        print(f"Step {step}/8: score={metrics['mean_score']:.2f} best={metrics['best_score']:.2f} ({elapsed:.1f}s)")
    
    # Test after training
    print()
    print("=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    print(f"\nQuestion: {test_q['question']}")
    sql = trainer.agent.generate_sql(test_q["question"])
    score, feedback = score_sql(test_q, sql)
    print(f"\nGenerated SQL:\n{sql}")
    print(f"\nScore: {score:.2f} | {', '.join(feedback)}")
    
    print()
    print("=" * 60)
    print("In production: Execute queries and score by correctness + performance")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

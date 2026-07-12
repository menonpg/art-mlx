#!/usr/bin/env python3
"""
ART-MLX: Code Review Agent

Train an AI to give better code review feedback by learning from
which reviews lead to accepted changes.

INDUSTRY: Developer Tools / DevOps
USE CASE: Automated PR reviews that are actually helpful

This example shows how to use GRPO to fine-tune an LLM to:
1. Identify real issues (not false positives)
2. Explain WHY something is a problem
3. Suggest concrete fixes
4. Match team coding standards

Usage:
    python examples/industry/code_review_agent.py
"""

import asyncio
import random
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
# Sample Code Diffs for Review
# ============================================================

SAMPLE_DIFFS = [
    {
        "filename": "auth.py",
        "diff": '''
@@ -45,6 +45,12 @@ def authenticate(username, password):
+def check_password(password):
+    if password == "admin123":
+        return True
+    return len(password) > 6
+
 def login(request):
     user = get_user(request.username)
+    if check_password(request.password):
+        return create_session(user)
''',
        "issues": ["hardcoded_password", "weak_validation", "no_hashing"],
        "severity": "critical",
    },
    {
        "filename": "api.py",
        "diff": '''
@@ -12,8 +12,15 @@ def get_users():
 def get_user(user_id):
-    return db.query(User).filter(User.id == user_id).first()
+    query = f"SELECT * FROM users WHERE id = {user_id}"
+    return db.execute(query)
''',
        "issues": ["sql_injection", "raw_query"],
        "severity": "critical",
    },
    {
        "filename": "utils.py",
        "diff": '''
@@ -1,5 +1,8 @@
+import time
+
 def process_data(items):
     results = []
     for item in items:
+        time.sleep(0.1)  # Rate limiting
         results.append(transform(item))
     return results
''',
        "issues": ["sleep_in_loop", "no_async"],
        "severity": "medium",
    },
    {
        "filename": "config.py",
        "diff": '''
@@ -5,3 +5,6 @@ DEBUG = True
 
+AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"
+DATABASE_URL = "postgres://admin:password123@prod.db.example.com/app"
''',
        "issues": ["exposed_secrets", "hardcoded_credentials"],
        "severity": "critical",
    },
    {
        "filename": "models.py",
        "diff": '''
@@ -20,6 +20,15 @@ class User:
 
+class Order:
+    def __init__(self):
+        self.items = []
+        self.total = 0
+    
+    def add_item(self, item):
+        self.items.append(item)
+        self.total = self.total + item.price
''',
        "issues": ["no_type_hints", "mutable_default"],
        "severity": "low",
    },
]


# ============================================================
# Review Scoring
# ============================================================

def score_review(diff_data: dict, review: str) -> tuple[float, list[str]]:
    """
    Score a code review. In production:
    - Track if review comments lead to changes
    - Measure false positive rate (comments dismissed)
    - Developer satisfaction surveys
    """
    score = 0.3  # Start with baseline
    feedback = []
    
    review_lower = review.lower()
    
    # Did it identify the actual issues?
    for issue in diff_data["issues"]:
        issue_keywords = {
            "hardcoded_password": ["hardcod", "password", "credential"],
            "weak_validation": ["validation", "weak", "security"],
            "no_hashing": ["hash", "plain", "encrypt"],
            "sql_injection": ["sql injection", "injection", "parameterized", "sanitize"],
            "raw_query": ["raw query", "orm", "prepared statement"],
            "sleep_in_loop": ["sleep", "blocking", "performance"],
            "no_async": ["async", "await", "concurrent"],
            "exposed_secrets": ["secret", "credential", "environment variable", "env var"],
            "hardcoded_credentials": ["hardcod", "credential", "secret"],
            "no_type_hints": ["type hint", "typing", "annotation"],
            "mutable_default": ["mutable", "default"],
        }
        
        keywords = issue_keywords.get(issue, [issue.replace("_", " ")])
        if any(kw in review_lower for kw in keywords):
            score += 0.15
            feedback.append(f"+identified_{issue}")
        else:
            feedback.append(f"-missed_{issue}")
    
    # Check for actionable suggestions
    if any(word in review_lower for word in ["should", "consider", "recommend", "suggest", "instead"]):
        score += 0.1
        feedback.append("+actionable")
    
    # Check for explanation of WHY
    if any(word in review_lower for word in ["because", "since", "this could", "risk", "vulnerability"]):
        score += 0.1
        feedback.append("+explains_why")
    
    # Severity-appropriate response
    if diff_data["severity"] == "critical":
        if any(word in review_lower for word in ["critical", "security", "block", "must fix", "vulnerability"]):
            score += 0.1
            feedback.append("+severity_match")
    
    # Penalize vague reviews
    if len(review) < 50:
        score -= 0.2
        feedback.append("-too_short")
    
    # Penalize nitpicking on critical issues
    if diff_data["severity"] == "critical":
        if "formatting" in review_lower or "whitespace" in review_lower:
            score -= 0.15
            feedback.append("-nitpicking")
    
    return min(max(score, 0), 1), feedback


# ============================================================
# Code Review Agent
# ============================================================

@dataclass
class ReviewAttempt:
    diff_data: dict
    review: str
    score: float
    feedback: list[str]


class CodeReviewAgent:
    """Agent that reviews code diffs."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = make_sampler(temp=0.6)
    
    def review_diff(self, diff_data: dict) -> str:
        """Generate a code review for a diff."""
        prompt = f"""You are a senior software engineer doing a code review.
Review this pull request diff and identify any issues.

FILE: {diff_data['filename']}
DIFF:
{diff_data['diff']}

Provide a helpful code review that:
1. Identifies specific issues (security, performance, bugs)
2. Explains WHY each issue matters
3. Suggests how to fix it
4. Is constructive and professional

YOUR REVIEW:"""

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=300,
            sampler=self.sampler,
            verbose=False
        )
        
        return response.strip()


# ============================================================
# GRPO Trainer
# ============================================================

class CodeReviewTrainer:
    """Train code review agent using GRPO."""
    
    def __init__(self, model, tokenizer, learning_rate: float = 1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(learning_rate=learning_rate)
        self.agent = CodeReviewAgent(model, tokenizer)
    
    def train_step(self, diffs: list[dict] = None, reviews_per_diff: int = 4) -> dict:
        """Train on a batch of diffs."""
        if diffs is None:
            diffs = random.sample(SAMPLE_DIFFS, min(2, len(SAMPLE_DIFFS)))
        
        all_attempts = []
        
        for diff_data in diffs:
            for _ in range(reviews_per_diff):
                review = self.agent.review_diff(diff_data)
                score, feedback = score_review(diff_data, review)
                all_attempts.append(ReviewAttempt(
                    diff_data=diff_data,
                    review=review,
                    score=score,
                    feedback=feedback,
                ))
        
        # GRPO advantages
        scores = [a.score for a in all_attempts]
        mean_score = sum(scores) / len(scores)
        
        total_loss = 0.0
        num_updates = 0
        
        for attempt in all_attempts:
            advantage = attempt.score - mean_score
            
            prompt = f"Review {attempt.diff_data['filename']}:"
            target = f" {attempt.review[:200]}"
            
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
    print("ART-MLX: Code Review Agent")
    print("=" * 60)
    print()
    
    model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading: {model_name}")
    
    start = time.time()
    model, tokenizer = load(model_name)
    print(f"✓ Loaded in {time.time() - start:.1f}s")
    
    model.freeze()
    linear_to_lora_layers(model, num_layers=8, config={"rank": 8, "scale": 20.0, "dropout": 0.0})
    num_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"✓ LoRA adapters: {num_trainable:,} trainable params")
    print()
    
    trainer = CodeReviewTrainer(model, tokenizer)
    
    # Test: SQL injection diff (critical)
    test_diff = SAMPLE_DIFFS[1]  # SQL injection
    
    print("=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    print(f"\nFile: {test_diff['filename']}")
    print(f"Diff: {test_diff['diff'][:200]}...")
    review = trainer.agent.review_diff(test_diff)
    score, feedback = score_review(test_diff, review)
    print(f"\nReview:\n{review[:400]}...")
    print(f"\nScore: {score:.2f} | {', '.join(feedback)}")
    
    # Training
    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for step in range(1, 9):
        start = time.time()
        metrics = trainer.train_step(reviews_per_diff=4)
        elapsed = time.time() - start
        print(f"Step {step}/8: score={metrics['mean_score']:.2f} best={metrics['best_score']:.2f} ({elapsed:.1f}s)")
    
    # After training
    print()
    print("=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    print(f"\nFile: {test_diff['filename']}")
    review = trainer.agent.review_diff(test_diff)
    score, feedback = score_review(test_diff, review)
    print(f"\nReview:\n{review[:400]}...")
    print(f"\nScore: {score:.2f} | {', '.join(feedback)}")
    
    print()
    print("=" * 60)
    print("In production: Score by whether developers accept/act on feedback")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

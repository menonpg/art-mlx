#!/usr/bin/env python3
"""
ART-MLX: Customer Support Email Agent

Train an AI to write better customer support responses by learning
from feedback on past responses.

INDUSTRY: Customer Service / SaaS
USE CASE: Auto-draft email replies that match company voice

This example shows how to use GRPO to fine-tune an LLM to:
1. Respond in the right tone (professional but friendly)
2. Address the actual issue mentioned
3. Offer concrete next steps
4. Keep responses concise

Usage:
    python examples/industry/email_support_agent.py
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
# Sample Customer Emails (In production, use your real tickets)
# ============================================================

SAMPLE_TICKETS = [
    {
        "subject": "Can't login to my account",
        "body": "I've been trying to login for the past hour but it keeps saying wrong password. I've tried resetting it 3 times. This is really frustrating, I have a deadline today.",
        "category": "account_access",
        "urgency": "high",
    },
    {
        "subject": "Billing question",
        "body": "Hi, I was charged $49 but I thought I was on the $29 plan. Can you explain?",
        "category": "billing",
        "urgency": "medium",
    },
    {
        "subject": "Feature request - dark mode",
        "body": "Would love to see dark mode added. Working late at night and the bright screen is rough on the eyes. Thanks!",
        "category": "feature_request",
        "urgency": "low",
    },
    {
        "subject": "Integration not working",
        "body": "Tried to connect Slack but getting an error. Screenshot attached. Need this working ASAP for our team.",
        "category": "technical",
        "urgency": "high",
    },
    {
        "subject": "Cancel subscription",
        "body": "Please cancel my subscription effective immediately.",
        "category": "churn",
        "urgency": "high",
    },
    {
        "subject": "Love the product!",
        "body": "Just wanted to say your tool has saved me hours every week. The new dashboard is great!",
        "category": "positive_feedback",
        "urgency": "low",
    },
]

# ============================================================
# Response Quality Scoring (Simulated human feedback)
# ============================================================

def score_response(ticket: dict, response: str) -> tuple[float, list[str]]:
    """
    Score a support response. In production, this would be:
    - Human ratings (1-5 stars)
    - Customer satisfaction surveys
    - Resolution rate tracking
    - Response time + reopened ticket rate
    
    For training, we simulate this with heuristics.
    """
    score = 0.5  # Start neutral
    feedback = []
    
    response_lower = response.lower()
    
    # Check for empathy/acknowledgment
    empathy_phrases = ["understand", "sorry", "apologize", "frustrating", "appreciate"]
    if any(phrase in response_lower for phrase in empathy_phrases):
        score += 0.1
        feedback.append("+empathy")
    
    # Check for concrete next steps
    action_phrases = ["click", "go to", "try", "steps", "here's how", "i'll", "we'll", "i've"]
    if any(phrase in response_lower for phrase in action_phrases):
        score += 0.15
        feedback.append("+action")
    
    # Check for professional greeting/closing
    if any(g in response_lower for g in ["hi", "hello", "dear"]):
        score += 0.05
        feedback.append("+greeting")
    if any(c in response_lower for c in ["best", "regards", "thanks", "let me know"]):
        score += 0.05
        feedback.append("+closing")
    
    # Penalize too short
    if len(response) < 100:
        score -= 0.2
        feedback.append("-too_short")
    
    # Penalize too long
    if len(response) > 800:
        score -= 0.1
        feedback.append("-too_long")
    
    # Category-specific scoring
    if ticket["category"] == "churn":
        if "offer" in response_lower or "discount" in response_lower or "help" in response_lower:
            score += 0.15
            feedback.append("+retention_attempt")
    
    if ticket["category"] == "billing":
        if "$" in response or "refund" in response_lower or "charge" in response_lower:
            score += 0.1
            feedback.append("+addressed_billing")
    
    if ticket["urgency"] == "high":
        if "immediately" in response_lower or "right away" in response_lower or "now" in response_lower:
            score += 0.1
            feedback.append("+urgency_acknowledged")
    
    return min(max(score, 0), 1), feedback


# ============================================================
# Email Agent
# ============================================================

@dataclass
class ResponseAttempt:
    ticket: dict
    response: str
    score: float
    feedback: list[str]


class EmailSupportAgent:
    """Agent that drafts customer support emails."""
    
    def __init__(self, model, tokenizer, company_name: str = "Acme Inc"):
        self.model = model
        self.tokenizer = tokenizer
        self.company_name = company_name
        self.sampler = make_sampler(temp=0.7)
    
    def draft_response(self, ticket: dict) -> str:
        """Generate a response to a customer ticket."""
        prompt = f"""You are a friendly customer support agent for {self.company_name}.
Write a helpful response to this customer email.

TICKET:
Subject: {ticket['subject']}
Body: {ticket['body']}

GUIDELINES:
- Be empathetic and acknowledge their concern
- Provide clear next steps or solutions
- Keep it concise (2-4 paragraphs)
- Professional but warm tone

YOUR RESPONSE:"""

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
    
    def process_ticket(self, ticket: dict) -> ResponseAttempt:
        """Process a ticket and score the response."""
        response = self.draft_response(ticket)
        score, feedback = score_response(ticket, response)
        
        return ResponseAttempt(
            ticket=ticket,
            response=response,
            score=score,
            feedback=feedback
        )


# ============================================================
# GRPO Trainer
# ============================================================

class EmailAgentTrainer:
    """Train the email agent using GRPO."""
    
    def __init__(self, model, tokenizer, learning_rate: float = 1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(learning_rate=learning_rate)
        self.agent = EmailSupportAgent(model, tokenizer)
    
    def train_step(self, tickets: list[dict] = None, num_responses: int = 4) -> dict:
        """
        Train on a batch of tickets.
        
        For each ticket, generate multiple responses, score them,
        and use GRPO to reinforce the better ones.
        """
        if tickets is None:
            tickets = random.sample(SAMPLE_TICKETS, min(2, len(SAMPLE_TICKETS)))
        
        all_attempts = []
        
        for ticket in tickets:
            # Generate multiple responses for the same ticket
            attempts = []
            for _ in range(num_responses):
                attempt = self.agent.process_ticket(ticket)
                attempts.append(attempt)
            all_attempts.extend(attempts)
        
        # Compute advantages relative to group mean
        scores = [a.score for a in all_attempts]
        mean_score = sum(scores) / len(scores)
        
        total_loss = 0.0
        num_updates = 0
        
        for attempt in all_attempts:
            advantage = attempt.score - mean_score
            
            # Create training example
            prompt = f"Ticket: {attempt.ticket['subject']}\nResponse:"
            target = f" {attempt.response[:200]}"  # Truncate for training
            
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
            "attempts": all_attempts,
        }


# ============================================================
# Main
# ============================================================

async def main():
    print("=" * 60)
    print("ART-MLX: Customer Support Email Agent")
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
    
    # Create trainer
    trainer = EmailAgentTrainer(model, tokenizer)
    
    # Show initial response
    print("=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    test_ticket = SAMPLE_TICKETS[0]
    print(f"\nTicket: {test_ticket['subject']}")
    print(f"Body: {test_ticket['body'][:100]}...")
    response = trainer.agent.draft_response(test_ticket)
    score, feedback = score_response(test_ticket, response)
    print(f"\nResponse:\n{response[:400]}...")
    print(f"\nScore: {score:.2f} | Feedback: {', '.join(feedback)}")
    
    # Training
    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    num_steps = 8
    for step in range(1, num_steps + 1):
        start = time.time()
        metrics = trainer.train_step(num_responses=4)
        elapsed = time.time() - start
        
        print(f"Step {step}/{num_steps}: "
              f"score={metrics['mean_score']:.2f} "
              f"best={metrics['best_score']:.2f} "
              f"loss={metrics['loss']:.4f} "
              f"({elapsed:.1f}s)")
    
    # Show final response
    print()
    print("=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    print(f"\nTicket: {test_ticket['subject']}")
    response = trainer.agent.draft_response(test_ticket)
    score, feedback = score_response(test_ticket, response)
    print(f"\nResponse:\n{response[:400]}...")
    print(f"\nScore: {score:.2f} | Feedback: {', '.join(feedback)}")
    
    print()
    print("=" * 60)
    print("Training complete! The agent should now write better support emails.")
    print("In production: replace score_response() with real customer feedback.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

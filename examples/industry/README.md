# Industry Examples

Real-world use cases for training AI agents with GRPO on Apple Silicon.

Each example shows:
1. A concrete business problem
2. How to define success (reward function)
3. Training loop with GRPO
4. Before/after comparison

## Examples

### 📧 Customer Support Email Agent
`email_support_agent.py`

Train an AI to draft better support responses by learning from customer satisfaction scores.

**Industry:** SaaS, E-commerce, Customer Service  
**Reward Signal:** Customer satisfaction ratings, resolution rate, response quality scores

```bash
python examples/industry/email_support_agent.py
```

---

### 🗄️ SQL Query Assistant
`sql_assistant.py`

Train an AI to write better SQL by learning from query execution results.

**Industry:** Data Analytics, BI Tools, Database Management  
**Reward Signal:** Query correctness, execution time, result accuracy

```bash
python examples/industry/sql_assistant.py
```

---

### 👀 Code Review Agent
`code_review_agent.py`

Train an AI to give more helpful code reviews by learning which feedback gets accepted.

**Industry:** Developer Tools, DevOps, Code Quality  
**Reward Signal:** Review acceptance rate, issue detection accuracy, developer satisfaction

```bash
python examples/industry/code_review_agent.py
```

---

## Building Your Own

The pattern is always:

```python
# 1. Define your task
class MyAgent:
    def do_task(self, input) -> output:
        # Use the LLM to complete the task
        ...

# 2. Define your reward
def score_output(input, output) -> float:
    # In production: real metrics (user ratings, success rate, etc.)
    # For training: heuristics that approximate real metrics
    ...

# 3. Train with GRPO
class MyTrainer:
    def train_step(self):
        # Generate multiple outputs
        outputs = [agent.do_task(input) for _ in range(N)]
        
        # Score them
        scores = [score_output(input, out) for out in outputs]
        
        # GRPO: reinforce outputs better than average
        mean = sum(scores) / len(scores)
        for output, score in zip(outputs, scores):
            advantage = score - mean
            # Update model weights with advantage-weighted loss
            ...
```

## Production Tips

1. **Start with heuristics, graduate to real feedback**
   - Heuristics: keyword matching, length checks, format validation
   - Real: A/B test responses, track user actions, collect ratings

2. **Collect diverse training data**
   - Sample across different categories/difficulties
   - Include edge cases

3. **Use appropriate model size**
   - 0.5B-3B for narrow tasks (games, simple agents)
   - 7B+ for complex reasoning (code review, support)

4. **Monitor for reward hacking**
   - If the model finds shortcuts that score high but aren't useful
   - Add more diverse reward signals

5. **Export and deploy**
   - `python -m art.mlx.export --checkpoint ./my_model --repo username/model --push`
   - Merge into base model or use LoRA adapters dynamically

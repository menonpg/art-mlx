# Industry Examples

Real-world use cases for training AI agents with GRPO on Apple Silicon.

Each example shows:
1. A concrete business problem
2. How to define success (reward function)
3. Training loop with GRPO
4. Before/after comparison

## What is the "training data" here? (read this first)

There is **no external dataset and no database**. Everything the model learns from is
written in plain text at the top of each script — about a page. Teaching the model works
like coaching a new hire with a **worked-examples sheet** and a **grading rubric**:

| Ingredient | What it is | Where it lives |
|---|---|---|
| **Example tasks** | A short hand-written list of realistic tasks (6 questions / 6 tickets / 5 diffs) | `SAMPLE_QUESTIONS` / `SAMPLE_TICKETS` / `SAMPLE_DIFFS` |
| **A grader (rubric)** | Plain rules that award points for what a good answer looks like | the `score_*()` function |

During training, for each example task the model writes several attempts, the grader scores
each, and training nudges the model toward the higher-scoring ones. Repeat 8 times.

**How a non-technical person does this:**
1. Collect ~5–10 **real** examples of the task (real tickets, real questions, real PRs) — copy-paste, no code.
2. Decide what a **good answer** looks like: simple rules, or have a colleague rate answers 1–5. This rubric matters most.
3. Paste your examples into the list at the top of the script (or hand them to a developer for ~15 min).
4. Run one command — the model practices and grades itself against your rubric, offline on your Mac.
5. Later, replace the rubric with a **real signal**: run the SQL and check the rows, track if the customer replied happy, or track if the developer accepted the review.

> One-sentence version: you teach by giving **a few example tasks** + **a rubric that scores answers**; the model practices against the rubric. You grade homework; the model is the student.

📊 **Full report with real training data samples and logs:** https://menonpg.github.io/art-mlx/mlx-business-examples.html

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

## Verified Results (2026-07-18)

All three examples were run end-to-end on an Apple M1 Mac (Python 3.13, MLX 0.32.0,
mlx-lm 0.31.3, Qwen2.5-0.5B-Instruct-4bit, ~1.47M trainable LoRA params). Each trained
through all 8 GRPO steps with real gradient updates.

| Example | Task | Before | Best in training | Step time |
|---|---|---|---|---|
| `sql_assistant.py` | Natural language → SQL | 0.90 | 0.90 | ~6–11s |
| `email_support_agent.py` | Support email drafting | 0.85 | 0.85 | ~12–18s |
| `code_review_agent.py` | PR code review | 0.80 | **1.00** | ~18–20s |

📊 **Full report with logs:** https://menonpg.github.io/art-mlx/mlx-business-examples.html

> **Honest caveat:** these runs prove the *training pipeline* works end-to-end on Apple
> Silicon — not that output quality reliably improves. With a tiny 0.5B model, a keyword
> heuristic reward, and only 8 steps, final outputs sometimes degrade (e.g. repetition
> loops). Production use needs a larger model, a genuine reward signal, and many more steps.
>
> **Tip:** run long jobs with `caffeinate -i python -u ...` so the Mac sleeping can't stall
> the GPU mid-run.

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

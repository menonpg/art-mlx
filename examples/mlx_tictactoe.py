"""
ART-MLX Tic Tac Toe — Full GRPO training with gradient updates.

This example trains a small model to play Tic Tac Toe using GRPO with
actual gradient updates via MLX.

Usage:
    python examples/mlx_tictactoe.py

Expected runtime: ~30-60 min on M1 Max with 0.5B model
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

# Check MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("ERROR: MLX not available. Install with: pip install mlx mlx-lm")
    exit(1)

import numpy as np


# =============================================================================
# Tic Tac Toe Game Logic
# =============================================================================

@dataclass
class TicTacToeState:
    """Game state for Tic Tac Toe."""
    board: list[str] = field(default_factory=lambda: [' '] * 9)
    current_player: Literal['X', 'O'] = 'X'
    
    @classmethod
    def new_game(cls) -> 'TicTacToeState':
        return cls()
    
    def display_with_numbers(self) -> str:
        b = [str(i) if self.board[i] == ' ' else self.board[i] for i in range(9)]
        return f" {b[0]} | {b[1]} | {b[2]} \n---+---+---\n {b[3]} | {b[4]} | {b[5]} \n---+---+---\n {b[6]} | {b[7]} | {b[8]} "
    
    def valid_moves(self) -> list[int]:
        return [i for i, cell in enumerate(self.board) if cell == ' ']
    
    def make_move(self, position: int) -> bool:
        if position < 0 or position > 8 or self.board[position] != ' ':
            return False
        self.board[position] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True
    
    def check_winner(self) -> str | None:
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != ' ':
                return self.board[line[0]]
        if ' ' not in self.board:
            return 'draw'
        return None


# =============================================================================
# Trajectory Storage
# =============================================================================

@dataclass
class Trajectory:
    """A single game trajectory with prompts, responses, and reward."""
    input_ids_list: list = field(default_factory=list)  # List of token sequences
    reward: float = 0.0


# =============================================================================
# GRPO Trainer with Actual Gradient Updates
# =============================================================================

class GRPOTrainer:
    """
    GRPO trainer with actual MLX gradient updates.
    
    This implements the core GRPO algorithm:
    1. Collect trajectories with rewards
    2. Compute advantages (relative to group mean)
    3. Compute policy gradient loss
    4. Backprop and update weights
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.clip_epsilon = clip_epsilon
        
        # Create optimizer for all trainable parameters
        self.optimizer = optim.AdamW(learning_rate=learning_rate)
        
        # Training state
        self.step_count = 0
        self.loss_history = []
        
        print(f"GRPOTrainer initialized with lr={learning_rate}")
    
    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """Compute GRPO advantages (relative to group mean, normalized)."""
        if not rewards:
            return []
        
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = variance ** 0.5 if variance > 0 else 1.0
        
        # Normalize advantages
        advantages = [(r - mean) / (std + 1e-8) for r in rewards]
        return advantages
    
    def compute_loss_for_sequence(
        self,
        model,
        input_ids: mx.array,
        advantage: float,
    ) -> mx.array:
        """
        Compute policy gradient loss for a single sequence.
        
        Loss = -advantage * mean(log_probs)
        
        For positive advantages, we minimize negative log likelihood.
        For negative advantages, we maximize it (discourage that behavior).
        """
        if input_ids.shape[0] < 2:
            return mx.array(0.0)
        
        # Forward pass - get logits
        # Input is all tokens except last, labels are all tokens except first
        inputs = input_ids[:-1].reshape(1, -1)
        labels = input_ids[1:].reshape(1, -1)
        
        logits = model(inputs)  # [1, seq_len-1, vocab_size]
        
        # Compute log probabilities
        log_probs = mx.log_softmax(logits, axis=-1)  # [1, seq_len-1, vocab_size]
        
        # Gather log probs for the actual tokens
        # labels shape: [1, seq_len-1]
        seq_len = labels.shape[1]
        vocab_size = log_probs.shape[-1]
        
        # Flatten for gathering
        log_probs_flat = log_probs.reshape(-1, vocab_size)  # [seq_len-1, vocab_size]
        labels_flat = labels.reshape(-1)  # [seq_len-1]
        
        # Get log prob of each actual token
        token_log_probs = mx.take_along_axis(
            log_probs_flat,
            labels_flat.reshape(-1, 1),
            axis=1
        ).squeeze(-1)  # [seq_len-1]
        
        # Mean log probability of the sequence
        mean_log_prob = mx.mean(token_log_probs)
        
        # Policy gradient loss: -advantage * log_prob
        # Positive advantage = we want to increase probability (minimize -log_prob)
        # Negative advantage = we want to decrease probability (maximize -log_prob)
        loss = -advantage * mean_log_prob
        
        return loss
    
    def train_step(self, trajectories: list[Trajectory]) -> dict:
        """
        One GRPO training step with actual gradient updates.
        
        Args:
            trajectories: List of trajectories with rewards
            
        Returns:
            Dictionary of training metrics
        """
        if not trajectories:
            return {"loss": 0.0, "num_updates": 0}
        
        # Compute advantages
        rewards = [t.reward for t in trajectories]
        advantages = self.compute_advantages(rewards)
        
        # Collect all (input_ids, advantage) pairs
        training_pairs = []
        for traj, adv in zip(trajectories, advantages):
            for input_ids in traj.input_ids_list:
                if len(input_ids) >= 2:
                    training_pairs.append((mx.array(input_ids), adv))
        
        if not training_pairs:
            return {"loss": 0.0, "num_updates": 0}
        
        total_loss = 0.0
        num_updates = 0
        
        # Process each sequence
        for input_ids, advantage in training_pairs:
            # Define loss function for this sequence
            def loss_fn(model):
                return self.compute_loss_for_sequence(model, input_ids, advantage)
            
            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(self.model)
            
            # Update model parameters
            self.optimizer.update(self.model, grads)
            
            # Force computation
            mx.eval(self.model.parameters())
            
            total_loss += float(loss)
            num_updates += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        
        self.step_count += 1
        self.loss_history.append(avg_loss)
        
        return {
            "loss": avg_loss,
            "num_updates": num_updates,
            "mean_reward": sum(rewards) / len(rewards),
            "mean_advantage": sum(abs(a) for a in advantages) / len(advantages),
            "step": self.step_count,
        }


# =============================================================================
# Agent
# =============================================================================

class TicTacToeAgent:
    """LLM-based Tic Tac Toe player that collects training data."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.current_trajectory = Trajectory()
    
    def reset(self):
        self.current_trajectory = Trajectory()
    
    def get_move(self, state: TicTacToeState) -> int:
        valid_moves = state.valid_moves()
        
        prompt = f"""You play Tic Tac Toe as X. Positions 0-8:
 0 | 1 | 2 
---+---+---
 3 | 4 | 5 
---+---+---
 6 | 7 | 8 

Board:
{state.display_with_numbers()}

Valid moves: {valid_moves}
Your move (single digit):"""

        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=3,
            temp=0.8,
            verbose=False
        )
        
        # Store tokenized sequence for training
        full_text = prompt + response
        tokens = self.tokenizer.encode(full_text)
        self.current_trajectory.input_ids_list.append(tokens)
        
        # Parse move
        for char in response:
            if char.isdigit():
                move = int(char)
                if move in valid_moves:
                    return move
        
        # Fallback to random
        return random.choice(valid_moves) if valid_moves else -1
    
    def get_trajectory(self) -> Trajectory:
        return self.current_trajectory


# =============================================================================
# Game Playing
# =============================================================================

def play_game(agent: TicTacToeAgent) -> tuple[str, Trajectory]:
    """Play one game, return (winner, trajectory)."""
    agent.reset()
    state = TicTacToeState.new_game()
    
    while True:
        # Agent's turn (X)
        move = agent.get_move(state)
        if not state.make_move(move):
            traj = agent.get_trajectory()
            traj.reward = -1.0
            return 'O', traj
        
        result = state.check_winner()
        if result:
            traj = agent.get_trajectory()
            traj.reward = 1.0 if result == 'X' else (0.0 if result == 'draw' else -1.0)
            return result, traj
        
        # Random opponent (O)
        opp_moves = state.valid_moves()
        if opp_moves:
            state.make_move(random.choice(opp_moves))
        
        result = state.check_winner()
        if result:
            traj = agent.get_trajectory()
            traj.reward = 1.0 if result == 'X' else (0.0 if result == 'draw' else -1.0)
            return result, traj


def evaluate(agent: TicTacToeAgent, num_games: int = 50) -> dict:
    """Evaluate agent win rate without training."""
    results = {'X': 0, 'O': 0, 'draw': 0}
    for _ in range(num_games):
        winner, _ = play_game(agent)
        results[winner] += 1
    return {
        'wins': results['X'],
        'losses': results['O'],
        'draws': results['draw'],
        'win_rate': results['X'] / num_games,
    }


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 60)
    print("ART-MLX Tic Tac Toe — GRPO Training")
    print("=" * 60)
    print()
    
    # Load model
    model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading: {model_name}")
    print("(First run downloads ~300MB)")
    
    start = time.time()
    model, tokenizer = load(model_name)
    print(f"✓ Loaded in {time.time() - start:.1f}s")
    print(f"✓ Device: {mx.default_device()}")
    print()
    
    # Create agent and trainer
    agent = TicTacToeAgent(model, tokenizer)
    trainer = GRPOTrainer(model, tokenizer, learning_rate=1e-5)
    
    # Initial evaluation
    print("Initial evaluation (30 games)...")
    initial = evaluate(agent, num_games=30)
    print(f"  Win rate: {initial['win_rate']:.1%}")
    print(f"  W/L/D: {initial['wins']}/{initial['losses']}/{initial['draws']}")
    print()
    
    # Training loop
    num_steps = 15
    games_per_step = 6
    
    print(f"Training: {num_steps} steps × {games_per_step} games = {num_steps * games_per_step} total games")
    print("-" * 60)
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Collect trajectories
        trajectories = []
        step_wins = 0
        
        for _ in range(games_per_step):
            winner, traj = play_game(agent)
            trajectories.append(traj)
            if winner == 'X':
                step_wins += 1
        
        # Train on trajectories
        metrics = trainer.train_step(trajectories)
        
        step_time = time.time() - step_start
        
        # Progress output
        print(f"Step {step+1:2d}/{num_steps}: "
              f"win_rate={step_wins/games_per_step:.0%} "
              f"loss={metrics['loss']:.4f} "
              f"updates={metrics['num_updates']} "
              f"({step_time:.1f}s)")
    
    print("-" * 60)
    print()
    
    # Final evaluation
    print("Final evaluation (30 games)...")
    final = evaluate(agent, num_games=30)
    print(f"  Win rate: {final['win_rate']:.1%}")
    print(f"  W/L/D: {final['wins']}/{final['losses']}/{final['draws']}")
    print()
    
    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    improvement = final['win_rate'] - initial['win_rate']
    print(f"Initial win rate: {initial['win_rate']:.1%}")
    print(f"Final win rate:   {final['win_rate']:.1%}")
    print(f"Change:           {improvement:+.1%}")
    print()
    
    if improvement > 0.1:
        print("✓ Significant improvement! GRPO training is working.")
    elif improvement > 0:
        print("~ Slight improvement. May need more training steps.")
    elif improvement > -0.1:
        print("~ No significant change. Model may need different hyperparameters.")
    else:
        print("⚠ Performance decreased. Check training setup.")
    
    print()
    print("Training complete. Loss history saved to memory.")
    print(f"Final loss values: {trainer.loss_history[-5:]}")


if __name__ == "__main__":
    asyncio.run(main())

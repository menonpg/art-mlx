#!/usr/bin/env python3
"""
ART-MLX: 2048 Game Training Example

Train a model to play 2048 using GRPO on Apple Silicon.
This is a more complex example than Tic Tac Toe — the game
requires strategic planning over many moves.

Usage:
    python examples/mlx_2048.py
"""

import asyncio
import random
import time
from dataclasses import dataclass

# Check MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available. Install with: pip install mlx mlx-lm")
    exit(1)


# ============================================================
# 2048 Game Logic
# ============================================================

WINNING_VALUE = 2048
BOARD_SIZE = 4


def create_board() -> list[list[int]]:
    """Create empty 4x4 board."""
    return [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def add_random_tile(board: list[list[int]]) -> bool:
    """Add a 2 or 4 to a random empty cell. Returns False if no space."""
    empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
    if not empty:
        return False
    i, j = random.choice(empty)
    board[i][j] = 4 if random.random() < 0.1 else 2
    return True


def new_game() -> list[list[int]]:
    """Start a new 2048 game with 2 random tiles."""
    board = create_board()
    add_random_tile(board)
    add_random_tile(board)
    return board


def slide_row_left(row: list[int]) -> tuple[list[int], int]:
    """Slide and merge a row to the left. Returns (new_row, points_earned)."""
    # Remove zeros
    tiles = [x for x in row if x != 0]
    
    # Merge adjacent equal tiles
    merged = []
    points = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged.append(tiles[i] * 2)
            points += tiles[i] * 2
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    
    # Pad with zeros
    return merged + [0] * (BOARD_SIZE - len(merged)), points


def rotate_board(board: list[list[int]], times: int = 1) -> list[list[int]]:
    """Rotate board 90 degrees clockwise, `times` times."""
    for _ in range(times % 4):
        board = [list(row) for row in zip(*board[::-1])]
    return board


def move(board: list[list[int]], direction: str) -> tuple[list[list[int]], int, bool]:
    """
    Apply a move to the board.
    Returns (new_board, points_earned, board_changed).
    """
    rotations = {"left": 0, "up": 1, "right": 2, "down": 3}
    if direction not in rotations:
        return board, 0, False
    
    # Rotate so we always slide left
    rotated = rotate_board([row[:] for row in board], rotations[direction])
    
    # Slide each row
    new_board = []
    total_points = 0
    for row in rotated:
        new_row, points = slide_row_left(row)
        new_board.append(new_row)
        total_points += points
    
    # Rotate back
    new_board = rotate_board(new_board, (4 - rotations[direction]) % 4)
    
    # Check if anything changed
    changed = new_board != board
    
    return new_board, total_points, changed


def get_max_tile(board: list[list[int]]) -> int:
    """Get the maximum tile value on the board."""
    return max(max(row) for row in board)


def can_move(board: list[list[int]]) -> bool:
    """Check if any moves are possible."""
    for direction in ["left", "right", "up", "down"]:
        _, _, changed = move(board, direction)
        if changed:
            return True
    return False


def render_board(board: list[list[int]]) -> str:
    """Render board as text for the model."""
    lines = []
    lines.append("┌────┬────┬────┬────┐")
    for i, row in enumerate(board):
        cells = []
        for val in row:
            if val == 0:
                cells.append("    ")
            else:
                cells.append(f"{val:4}")
        lines.append("│" + "│".join(cells) + "│")
        if i < BOARD_SIZE - 1:
            lines.append("├────┼────┼────┼────┤")
    lines.append("└────┴────┴────┴────┘")
    return "\n".join(lines)


# ============================================================
# Agent
# ============================================================

@dataclass
class GameResult:
    """Result of a single game."""
    max_tile: int
    total_score: int
    num_moves: int
    won: bool
    trajectory: list[tuple[str, str, str]]  # (board_state, move, result)


class Game2048Agent:
    """Agent that plays 2048 using an LLM."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = make_sampler(temp=0.7)
    
    def get_move(self, board: list[list[int]], history: list[str] = None) -> str:
        """Get the next move from the model."""
        board_str = render_board(board)
        max_tile = get_max_tile(board)
        
        prompt = f"""You are an expert 2048 player. Your goal is to reach 2048 by combining tiles strategically.

Current board (max tile: {max_tile}):
{board_str}

Valid moves: left, right, up, down

Strategy tips:
- Keep your highest tile in a corner
- Build a chain of decreasing values
- Avoid moving the high tile out of the corner

Respond with ONLY one word: left, right, up, or down."""

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=5,
            sampler=self.sampler,
            verbose=False
        )
        
        # Parse the move
        response_lower = response.lower().strip()
        for m in ["left", "right", "up", "down"]:
            if m in response_lower:
                return m
        
        # Default to a random valid move if parsing fails
        return random.choice(["left", "right", "up", "down"])
    
    def play_game(self, max_moves: int = 500) -> GameResult:
        """Play a complete game of 2048."""
        board = new_game()
        total_score = 0
        num_moves = 0
        trajectory = []
        
        while num_moves < max_moves:
            if not can_move(board):
                break
            
            board_str = render_board(board)
            move_choice = self.get_move(board)
            
            new_board, points, changed = move(board, move_choice)
            
            if changed:
                board = new_board
                total_score += points
                add_random_tile(board)
                trajectory.append((board_str, move_choice, f"+{points}"))
                num_moves += 1
            else:
                # Invalid move, try again (doesn't count)
                trajectory.append((board_str, move_choice, "invalid"))
        
        max_tile = get_max_tile(board)
        won = max_tile >= WINNING_VALUE
        
        return GameResult(
            max_tile=max_tile,
            total_score=total_score,
            num_moves=num_moves,
            won=won,
            trajectory=trajectory
        )


# ============================================================
# GRPO Trainer
# ============================================================

class GRPO2048Trainer:
    """GRPO trainer for 2048."""
    
    def __init__(self, model, tokenizer, learning_rate: float = 1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(learning_rate=learning_rate)
        self.agent = Game2048Agent(model, tokenizer)
    
    def compute_reward(self, result: GameResult) -> float:
        """Compute reward based on game result."""
        if result.won:
            return 2.0  # Big bonus for winning
        
        # Log-scale reward based on max tile (2→0, 2048→1)
        import math
        max_tile_reward = (math.log2(result.max_tile) - 1) / 10  # 0 to ~1
        
        # Bonus for high scores
        score_reward = min(result.total_score / 10000, 0.5)
        
        return max_tile_reward + score_reward
    
    def train_step(self, num_games: int = 4) -> dict:
        """Run one training step with multiple games."""
        # Play games and collect results
        results = []
        for _ in range(num_games):
            result = self.agent.play_game(max_moves=200)
            results.append(result)
        
        # Compute rewards and advantages
        rewards = [self.compute_reward(r) for r in results]
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]
        
        # For now, we'll update based on the final move of each game
        # (A full implementation would update on all moves)
        total_loss = 0.0
        num_updates = 0
        
        for result, advantage in zip(results, advantages):
            if not result.trajectory:
                continue
            
            # Get the last few moves
            for board_str, move_choice, _ in result.trajectory[-5:]:
                prompt = f"Board:\n{board_str}\nBest move:"
                target = f" {move_choice}"
                
                # Tokenize
                full_text = prompt + target
                tokens = self.tokenizer.encode(full_text)
                if len(tokens) < 2:
                    continue
                
                input_ids = mx.array([tokens[:-1]])
                labels = mx.array([tokens[1:]])
                
                def loss_fn():
                    logits = self.model(input_ids)
                    log_probs = nn.log_softmax(logits, axis=-1)
                    
                    # Gather log probs for target tokens
                    target_log_probs = mx.take_along_axis(
                        log_probs[0], labels[0, :, None], axis=-1
                    ).squeeze(-1)
                    
                    # Policy gradient loss (negative because we maximize)
                    return -mx.mean(target_log_probs) * advantage
                
                loss, grads = nn.value_and_grad(self.model, loss_fn)()
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
                
                total_loss += float(loss)
                num_updates += 1
        
        # Compute stats
        max_tiles = [r.max_tile for r in results]
        scores = [r.total_score for r in results]
        
        return {
            "loss": total_loss / max(num_updates, 1),
            "mean_reward": mean_reward,
            "max_tile_avg": sum(max_tiles) / len(max_tiles),
            "max_tile_best": max(max_tiles),
            "score_avg": sum(scores) / len(scores),
            "num_updates": num_updates
        }


# ============================================================
# Main
# ============================================================

async def main():
    print("=" * 60)
    print("ART-MLX 2048 — GRPO Training")
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
    
    # Apply LoRA
    model.freeze()
    lora_config = {"rank": 8, "scale": 20.0, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers=8, config=lora_config)
    num_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"✓ LoRA adapters applied ({num_trainable:,} trainable params)")
    print()
    
    # Create trainer
    trainer = GRPO2048Trainer(model, tokenizer, learning_rate=1e-5)
    
    # Initial evaluation
    print("Initial evaluation (3 games)...")
    results = [trainer.agent.play_game(max_moves=200) for _ in range(3)]
    max_tiles = [r.max_tile for r in results]
    scores = [r.total_score for r in results]
    print(f"  Max tiles: {max_tiles}")
    print(f"  Best: {max(max_tiles)}, Avg score: {sum(scores)/len(scores):.0f}")
    print()
    
    # Training
    num_steps = 10
    games_per_step = 4
    print(f"Training: {num_steps} steps × {games_per_step} games = {num_steps * games_per_step} total games")
    print("-" * 60)
    
    for step in range(1, num_steps + 1):
        start = time.time()
        metrics = trainer.train_step(num_games=games_per_step)
        elapsed = time.time() - start
        
        print(f"Step {step}/{num_steps}: "
              f"max_tile={metrics['max_tile_best']:4} "
              f"avg={metrics['max_tile_avg']:.0f} "
              f"score={metrics['score_avg']:.0f} "
              f"loss={metrics['loss']:.4f} "
              f"({elapsed:.1f}s)")
    
    print("-" * 60)
    print()
    
    # Final evaluation
    print("Final evaluation (3 games)...")
    results = [trainer.agent.play_game(max_moves=200) for _ in range(3)]
    max_tiles = [r.max_tile for r in results]
    scores = [r.total_score for r in results]
    print(f"  Max tiles: {max_tiles}")
    print(f"  Best: {max(max_tiles)}, Avg score: {sum(scores)/len(scores):.0f}")
    print()
    
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

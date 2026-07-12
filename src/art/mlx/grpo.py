"""
GRPO (Group Relative Policy Optimization) implementation for MLX.

This is the core RL algorithm that makes ART work. GRPO is similar to PPO
but uses group-relative advantages instead of a learned value function.

Key insight: Within a group of rollouts on the same task, compare trajectories
to each other rather than to an absolute baseline. This provides a stable
training signal without needing a critic network.

Reference: DeepSeek-R1 paper uses GRPO for reasoning model training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Learning rate
    learning_rate: float = 1e-5
    
    # Clip range for policy ratio (like PPO)
    clip_epsilon: float = 0.2
    
    # KL divergence penalty coefficient
    kl_coef: float = 0.1
    
    # Entropy bonus coefficient (encourages exploration)
    entropy_coef: float = 0.01
    
    # Number of optimization epochs per batch
    num_epochs: int = 4
    
    # Mini-batch size for gradient updates
    mini_batch_size: int = 4
    
    # Maximum gradient norm for clipping
    max_grad_norm: float = 1.0
    
    # Whether to normalize advantages
    normalize_advantages: bool = True
    
    # Minimum group size for training (skip smaller groups)
    min_group_size: int = 2


@dataclass 
class GRPOBatch:
    """A batch of tokenized trajectories for GRPO training."""
    
    # Input token IDs [batch, seq_len]
    input_ids: Any  # mx.array
    
    # Attention mask [batch, seq_len]  
    attention_mask: Any  # mx.array
    
    # Labels for loss computation [batch, seq_len]
    labels: Any  # mx.array
    
    # Per-trajectory advantages [batch]
    advantages: Any  # mx.array
    
    # Old log probabilities (for importance sampling) [batch]
    old_log_probs: Any  # mx.array
    
    # Reference log probabilities (for KL penalty) [batch]
    ref_log_probs: Any  # mx.array


class GRPOTrainer:
    """
    GRPO trainer for MLX models.
    
    Implements the Group Relative Policy Optimization algorithm:
    
    1. Collect trajectories in groups (same task, multiple rollouts)
    2. Compute advantages relative to group mean
    3. Update policy to increase probability of high-advantage actions
    4. Apply KL penalty to prevent too much deviation from reference
    
    Example:
        >>> trainer = GRPOTrainer(model, tokenizer, config)
        >>> for groups in trajectory_batches:
        ...     metrics = trainer.train_step(groups)
        ...     print(f"Loss: {metrics['loss']:.4f}")
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: GRPOConfig | None = None,
        reference_model: Any | None = None,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            model: The MLX model to train (with LoRA adapters)
            tokenizer: The tokenizer for the model
            config: GRPO configuration
            reference_model: Optional frozen reference model for KL penalty.
                           If None, uses the initial model state.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.reference_model = reference_model
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate
        )
        
        # Training state
        self.step = 0
        self._metrics_history: list[dict] = []
    
    def compute_advantages(
        self,
        rewards: list[float],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute GRPO advantages from rewards.
        
        GRPO advantage = reward - mean(rewards_in_group)
        
        This is the key difference from PPO: we don't need a value function,
        just compare trajectories within the same group.
        
        Args:
            rewards: List of trajectory rewards in a group
            normalize: Whether to normalize by standard deviation
        
        Returns:
            Array of advantages
        """
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        
        advantages = rewards - mean_reward
        
        if normalize and len(rewards) > 1:
            std = np.std(rewards)
            if std > 1e-8:
                advantages = advantages / std
        
        return advantages
    
    def compute_log_probs(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        labels: Any,
    ) -> Any:
        """
        Compute log probabilities of the labels under the model.
        
        Args:
            model: The language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs
        
        Returns:
            Log probabilities [batch_size]
        """
        # Forward pass
        logits = model(input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        # Compute log softmax
        log_probs = mx.log_softmax(shift_logits, axis=-1)
        
        # Gather log probs for actual tokens
        batch_size, seq_len, vocab_size = log_probs.shape
        
        # Create indices for gathering
        batch_indices = mx.arange(batch_size)[:, None]
        seq_indices = mx.arange(seq_len)[None, :]
        
        # Gather and sum (masked)
        token_log_probs = log_probs[batch_indices, seq_indices, shift_labels]
        
        # Mask padding
        mask = (shift_labels != self.tokenizer.pad_token_id).astype(mx.float32)
        token_log_probs = token_log_probs * mask
        
        # Sum over sequence
        sequence_log_probs = mx.sum(token_log_probs, axis=-1)
        
        return sequence_log_probs
    
    def grpo_loss(
        self,
        model: Any,
        batch: GRPOBatch,
    ) -> tuple[Any, dict]:
        """
        Compute the GRPO loss.
        
        Loss = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)] + kl_coef * KL + entropy_coef * H
        
        Where:
        - r = exp(log_prob - old_log_prob) is the importance sampling ratio
        - A is the advantage
        - KL is the KL divergence from reference policy
        - H is the entropy bonus
        
        Args:
            model: The policy model
            batch: Batch of training data
        
        Returns:
            (loss, metrics_dict)
        """
        # Compute current log probs
        log_probs = self.compute_log_probs(
            model,
            batch.input_ids,
            batch.attention_mask,
            batch.labels
        )
        
        # Importance sampling ratio
        ratio = mx.exp(log_probs - batch.old_log_probs)
        
        # Clipped ratio
        clipped_ratio = mx.clip(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon
        )
        
        # Policy loss (negative because we maximize)
        policy_loss_1 = ratio * batch.advantages
        policy_loss_2 = clipped_ratio * batch.advantages
        policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
        
        # KL divergence penalty
        kl_div = mx.mean(batch.ref_log_probs - log_probs)
        kl_loss = self.config.kl_coef * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_loss
        
        metrics = {
            "policy_loss": float(policy_loss),
            "kl_div": float(kl_div),
            "kl_loss": float(kl_loss),
            "total_loss": float(total_loss),
            "ratio_mean": float(mx.mean(ratio)),
            "ratio_std": float(mx.std(ratio)),
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        trajectory_groups: list[Any],  # list[TrajectoryGroup]
    ) -> dict[str, float]:
        """
        Perform one GRPO training step on trajectory groups.
        
        Args:
            trajectory_groups: Groups of trajectories with rewards
        
        Returns:
            Dictionary of training metrics
        """
        # Prepare batches from trajectory groups
        batches = self._prepare_batches(trajectory_groups)
        
        if not batches:
            logger.warning("No valid batches to train on")
            return {"loss": 0.0, "num_batches": 0}
        
        total_metrics: dict[str, float] = {}
        num_updates = 0
        
        # Multiple epochs over the data
        for epoch in range(self.config.num_epochs):
            for batch in batches:
                # Compute loss and gradients
                loss_and_grad_fn = mx.value_and_grad(
                    lambda m: self.grpo_loss(m, batch)[0]
                )
                loss, grads = loss_and_grad_fn(self.model)
                
                # Gradient clipping
                grads = self._clip_gradients(grads)
                
                # Optimizer step
                self.optimizer.update(self.model, grads)
                
                # Accumulate metrics
                _, step_metrics = self.grpo_loss(self.model, batch)
                for k, v in step_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                num_updates += 1
        
        # Average metrics
        if num_updates > 0:
            for k in total_metrics:
                total_metrics[k] /= num_updates
        
        total_metrics["num_updates"] = num_updates
        total_metrics["step"] = self.step
        
        self.step += 1
        self._metrics_history.append(total_metrics)
        
        return total_metrics
    
    def _prepare_batches(
        self,
        trajectory_groups: list[Any],
    ) -> list[GRPOBatch]:
        """Convert trajectory groups into training batches."""
        batches = []
        
        for group in trajectory_groups:
            trajectories = group.trajectories
            
            # Filter trajectories with rewards
            valid_trajectories = [
                t for t in trajectories 
                if t.reward is not None
            ]
            
            if len(valid_trajectories) < self.config.min_group_size:
                continue
            
            # Compute advantages
            rewards = [t.reward for t in valid_trajectories]
            advantages = self.compute_advantages(
                rewards, 
                normalize=self.config.normalize_advantages
            )
            
            # Tokenize trajectories
            # TODO: Implement proper tokenization
            # For now, this is a placeholder
            
        return batches
    
    def _clip_gradients(self, grads: Any) -> Any:
        """Clip gradients by global norm."""
        # Compute global norm
        total_norm_sq = 0.0
        for g in grads.values():
            if g is not None:
                total_norm_sq += mx.sum(g ** 2)
        total_norm = mx.sqrt(total_norm_sq)
        
        # Clip if needed
        clip_coef = self.config.max_grad_norm / (total_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, 1.0)
        
        clipped_grads = {}
        for k, g in grads.items():
            if g is not None:
                clipped_grads[k] = g * clip_coef
            else:
                clipped_grads[k] = g
        
        return clipped_grads
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        # TODO: Implement checkpoint saving
        pass
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        # TODO: Implement checkpoint loading
        pass

"""
ART-MLX Quick Test — Verify MLX backend works on your Mac.

Run this script to verify your environment is set up correctly:

    python examples/mlx_quicktest.py

Requirements:
    pip install mlx mlx-lm

Expected output:
    ✓ MLX available
    ✓ MLX-LM available
    ✓ Device: gpu (or cpu on Intel)
    ✓ Model loaded: mlx-community/Qwen2.5-0.5B-Instruct-4bit
    ✓ Generation working
    ✓ Ready for GRPO training!
"""

import sys


def check_mlx():
    """Check MLX availability."""
    try:
        import mlx.core as mx
        print(f"✓ MLX available (version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'})")
        print(f"✓ Device: {mx.default_device()}")
        return True
    except ImportError as e:
        print(f"✗ MLX not available: {e}")
        print("  Install with: pip install mlx")
        return False


def check_mlx_lm():
    """Check MLX-LM availability."""
    try:
        from mlx_lm import load, generate
        print("✓ MLX-LM available")
        return True
    except ImportError as e:
        print(f"✗ MLX-LM not available: {e}")
        print("  Install with: pip install mlx-lm")
        return False


def test_model_loading():
    """Test loading a small model."""
    from mlx_lm import load, generate
    
    # Use a tiny model for quick testing
    model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    
    print(f"\nLoading {model_name}...")
    try:
        model, tokenizer = load(model_name)
        print(f"✓ Model loaded: {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None


def test_generation(model, tokenizer):
    """Test generation works."""
    from mlx_lm import generate
    
    prompt = "What is 2 + 2?"
    
    print(f"\nTesting generation with prompt: '{prompt}'")
    try:
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
            max_tokens=50,
            verbose=False
        )
        print(f"✓ Generation working")
        print(f"  Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


def test_art_mlx_import():
    """Test ART MLX backend import."""
    try:
        from art.mlx import MLXBackend
        print("✓ ART MLX backend importable")
        return True
    except ImportError as e:
        print(f"✗ ART MLX backend not importable: {e}")
        print("  Make sure you installed ART with: pip install -e .")
        return False


def main():
    print("=" * 50)
    print("ART-MLX Environment Check")
    print("=" * 50)
    print()
    
    # Check dependencies
    if not check_mlx():
        sys.exit(1)
    
    if not check_mlx_lm():
        sys.exit(1)
    
    # Test model loading and generation
    model, tokenizer = test_model_loading()
    if model is None:
        sys.exit(1)
    
    if not test_generation(model, tokenizer):
        sys.exit(1)
    
    # Test ART import
    test_art_mlx_import()
    
    print()
    print("=" * 50)
    print("✓ Ready for GRPO training!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Run the Tic Tac Toe example:")
    print("   python examples/mlx_tictactoe.py")
    print()
    print("2. Or try the 2048 example:")
    print("   python examples/mlx_2048.py")


if __name__ == "__main__":
    main()

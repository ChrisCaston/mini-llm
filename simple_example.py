"""
Simple example showing how to use the Mini LLM for training and inference.
This is a minimal example to get you started quickly.
"""

import torch
from mini_llm import MiniLLM, SimpleTokenizer
import json

def quick_training_example():
    """Minimal training example with very small data."""
    
    # Simple training data
    texts = [
        "hello world this is a simple example",
        "machine learning is very interesting",
        "deep learning models can learn patterns",
        "artificial intelligence is the future",
        "transformers are powerful neural networks",
        "attention mechanisms help models focus",
        "language models generate coherent text",
        "python is great for machine learning"
    ] * 20  # Repeat for more training examples
    
    # Build tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create a small model for quick training
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,      # Small model
        num_heads=4,     # Fewer heads
        num_layers=3,    # Fewer layers
        d_ff=128,        # Smaller FFN
        max_seq_length=64,
        dropout=0.1
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Simple training loop (very basic)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Convert texts to sequences
    sequences = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 10:  # Minimum sequence length
            sequences.append(tokens[:32])  # Truncate to max length
    
    print(f"Training on {len(sequences)} sequences...")
    
    # Quick training (just a few steps for demo)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for seq in sequences:
            if len(seq) < 2:
                continue
                
            # Prepare input and target
            input_seq = torch.tensor(seq[:-1]).unsqueeze(0)
            target_seq = torch.tensor(seq[1:]).unsqueeze(0)
            
            # Forward pass
            logits = model(input_seq)
            loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(sequences)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Save model and tokenizer
    torch.save(model.state_dict(), 'simple_model.pt')
    tokenizer.save('simple_tokenizer.json')
    
    return model, tokenizer

def quick_inference_example(model, tokenizer):
    """Simple text generation example."""
    
    prompts = ["hello", "machine", "deep learning"]
    
    model.eval()
    print("\n" + "="*40)
    print("Generated Text Examples")
    print("="*40)
    
    for prompt in prompts:
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens])
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=10
            )
        
        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")

def load_and_test():
    """Load saved model and test it."""
    try:
        # Load tokenizer
        tokenizer = SimpleTokenizer()
        tokenizer.load('simple_tokenizer.json')
        
        # Load model
        model = MiniLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=3,
            d_ff=128,
            max_seq_length=64,
            dropout=0.1
        )
        model.load_state_dict(torch.load('simple_model.pt', map_location='cpu'))
        
        print("Loaded saved model successfully!")
        quick_inference_example(model, tokenizer)
        
    except FileNotFoundError:
        print("No saved model found. Training new model...")
        model, tokenizer = quick_training_example()
        quick_inference_example(model, tokenizer)

def interactive_demo():
    """Simple interactive demo."""
    try:
        # Load model
        tokenizer = SimpleTokenizer()
        tokenizer.load('simple_tokenizer.json')
        
        model = MiniLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=3,
            d_ff=128,
            max_seq_length=64,
            dropout=0.1
        )
        model.load_state_dict(torch.load('simple_model.pt', map_location='cpu'))
        model.eval()
        
        print("\n" + "="*50)
        print("Mini LLM Interactive Demo")
        print("="*50)
        print("Enter text prompts (or 'quit' to exit):")
        
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() == 'quit':
                break
                
            if not prompt:
                continue
                
            # Generate
            prompt_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_tokens])
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=30,
                    temperature=0.7,
                    top_k=15
                )
            
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"Generated: {generated_text}")
            
    except FileNotFoundError:
        print("No saved model found. Please run training first!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        quick_training_example()
        print("\nTraining completed! Run with 'interactive' to test.")
    else:
        print("Mini LLM Simple Example")
        print("Usage:")
        print("  python simple_example.py train       # Train a small model")
        print("  python simple_example.py interactive # Interactive generation")
        print("  python simple_example.py             # Load or train + demo")
        print()
        load_and_test()

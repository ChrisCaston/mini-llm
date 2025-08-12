import torch
import argparse
from mini_llm import MiniLLM, SimpleTokenizer

def load_model(model_path, tokenizer_path, device='cpu'):
    """Load trained model and tokenizer."""
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Initialize model with same config as training
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_seq_length=256,
        dropout=0.1
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40, device='cpu'):
    """Generate text from a prompt."""
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    return generated_text

def interactive_mode(model, tokenizer, device):
    """Interactive text generation mode."""
    print("="*60)
    print("Mini LLM Interactive Mode")
    print("="*60)
    print("Enter your prompts below. Type 'quit' to exit.")
    print("Type 'settings' to adjust generation parameters.")
    print("-"*60)
    
    # Default settings
    max_length = 50
    temperature = 0.8
    top_k = 40
    
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'settings':
                print(f"\nCurrent settings:")
                print(f"  Max length: {max_length}")
                print(f"  Temperature: {temperature}")
                print(f"  Top-k: {top_k}")
                
                try:
                    new_max = input(f"Max length ({max_length}): ").strip()
                    if new_max:
                        max_length = int(new_max)
                    
                    new_temp = input(f"Temperature ({temperature}): ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    new_k = input(f"Top-k ({top_k}): ").strip()
                    if new_k:
                        top_k = int(new_k)
                    
                    print("Settings updated!")
                except ValueError:
                    print("Invalid input. Settings unchanged.")
                
                continue
            
            if not user_input:
                continue
            
            print(f"\nGenerating text for: '{user_input}'")
            print("-" * 40)
            
            generated = generate_text(
                model, tokenizer, user_input,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            
            print(f"Generated: {generated}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def batch_generation(model, tokenizer, prompts, max_length, temperature, top_k, device):
    """Generate text for multiple prompts."""
    print("="*60)
    print("Batch Text Generation")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
        print("-" * 40)
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            device=device
        )
        
        print(f"Generated: {generated}")

def main():
    parser = argparse.ArgumentParser(description='Mini LLM Inference')
    parser.add_argument('--model', type=str, default='mini_llm_final.pt',
                       help='Path to trained model')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt for generation')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                       help='Multiple prompts for batch generation')
    parser.add_argument('--max_length', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_model(args.model, args.tokenizer, device)
        print("Model loaded successfully!")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Choose mode
        if args.interactive:
            interactive_mode(model, tokenizer, device)
        elif args.prompts:
            batch_generation(
                model, tokenizer, args.prompts,
                args.max_length, args.temperature, args.top_k, device
            )
        elif args.prompt:
            print(f"\nPrompt: '{args.prompt}'")
            print("-" * 40)
            generated = generate_text(
                model, tokenizer, args.prompt,
                args.max_length, args.temperature, args.top_k, device
            )
            print(f"Generated: {generated}")
        else:
            # Default: run a few sample generations
            sample_prompts = [
                "the quick brown",
                "machine learning is",
                "hello world",
                "artificial intelligence"
            ]
            batch_generation(
                model, tokenizer, sample_prompts,
                args.max_length, args.temperature, args.top_k, device
            )
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Make sure you have trained the model first!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
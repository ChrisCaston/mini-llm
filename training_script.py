import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mini_llm import MiniLLM, SimpleTokenizer, train_step

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize all texts
        self.tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.tokens.extend(tokens)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):
            seq = self.tokens[i:i + seq_length + 1]  # +1 for target
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

def prepare_data(data_path, tokenizer_path=None):
    """Prepare training data."""
    # Load your text data here
    # For demonstration, using some sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a simple example of text generation.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models can learn complex patterns in data.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized the field of NLP.",
        "Attention is all you need for sequence to sequence learning.",
        "Language models can generate coherent and contextual text.",
    ]
    
    # In practice, load from files:
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     texts = f.readlines()
    
    texts = sample_texts * 100  # Repeat for more training data
    
    # Build tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    
    if tokenizer_path:
        tokenizer.save(tokenizer_path)
    
    return texts, tokenizer

def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=10,
    lr=1e-3,
    device='cpu',
    save_path='model_checkpoint.pt'
):
    """Training loop for the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            batch = batch.to(device)
            loss = train_step(model, batch, optimizer, criterion)
            total_train_loss += loss
            train_steps += 1
            
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]
                    
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    total_val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = total_val_loss / val_steps
            val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}')
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if val_loader else None,
            }, f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

def generate_text_samples(model, tokenizer, prompts, device='cpu', max_length=100):
    """Generate text samples using the trained model."""
    model.eval()
    model.to(device)
    
    print("\n" + "="*50)
    print("Generated Text Samples")
    print("="*50)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        print("-" * 30)
        
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=0.8,
                top_k=40
            )
        
        # Decode and print
        generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        print(f"Generated: '{generated_text}'")

def main():
    """Main training script."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    config = {
        'vocab_size': None,  # Will be set after tokenizer
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 512,
        'max_seq_length': 256,
        'dropout': 0.1,
        'seq_length': 128,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 20,
    }
    
    # Prepare data
    print("Preparing data...")
    texts, tokenizer = prepare_data('data.txt', 'tokenizer.json')
    config['vocab_size'] = tokenizer.vocab_size
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of texts: {len(texts)}")
    
    # Create datasets
    dataset = TextDataset(texts, tokenizer, config['seq_length'])
    
    # Split into train/validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = MiniLLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        device=device,
        save_path='mini_llm_final.pt'
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Generate some sample text
    sample_prompts = [
        "hello",
        "the",
        "machine learning",
        "artificial intelligence"
    ]
    
    generate_text_samples(model, tokenizer, sample_prompts, device)
    
    print("\nTraining completed!")
    print("Model saved as 'mini_llm_final.pt'")
    print("Tokenizer saved as 'tokenizer.json'")

if __name__ == "__main__":
    main()

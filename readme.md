# Mini LLM - A Simple Language Model Implementation

A lightweight implementation of a transformer-based language model built from scratch using PyTorch. This project demonstrates the core concepts behind Large Language Models (LLMs) in an educational and accessible way.

## Features

- **Pure PyTorch Implementation**: Built from scratch without relying on high-level libraries
- **Transformer Architecture**: Includes multi-head attention, feed-forward networks, and positional encoding
- **Character-level Tokenization**: Simple tokenizer for getting started quickly
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Text Generation**: Support for various sampling strategies (temperature, top-k)
- **Interactive Mode**: Real-time text generation interface
- **Configurable**: Easy to modify model size and hyperparameters

## Model Architecture

```
Mini LLM Architecture:
├── Token Embedding (vocab_size → d_model)
├── Positional Encoding
├── N × Transformer Blocks:
│   ├── Multi-Head Self-Attention
│   ├── Layer Normalization
│   ├── Feed-Forward Network
│   └── Residual Connections
├── Final Layer Normalization
└── Output Projection (d_model → vocab_size)
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/mini-llm.git
cd mini-llm
pip install -r requirements.txt
```

### Training

1. **Prepare your data**: Place your training text in a file or modify the sample data in `training_script.py`

2. **Train the model**:
```bash
python training_script.py
```

3. **Monitor training**: The script will save checkpoints and generate training curves

### Inference

**Interactive mode**:
```bash
python inference_script.py --interactive
```

**Single prompt**:
```bash
python inference_script.py --prompt "The future of AI is"
```

**Batch generation**:
```bash
python inference_script.py --prompts "Hello world" "Machine learning" "Deep learning"
```

**Custom settings**:
```bash
python inference_script.py --prompt "Once upon a time" --max_length 100 --temperature 0.9 --top_k 50
```

## Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Hidden dimension size |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of transformer blocks |
| `d_ff` | 512 | Feed-forward network dimension |
| `max_seq_length` | 256 | Maximum sequence length |
| `dropout` | 0.1 | Dropout rate |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `epochs` | 20 | Number of training epochs |
| `seq_length` | 128 | Training sequence length |

## File Structure

```
mini-llm/
├── mini_llm.py           # Core model implementation
├── training_script.py    # Training pipeline
├── inference_script.py   # Text generation interface  
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── examples/            # Example usage scripts
    ├── simple_training.py
    └── custom_data.py
```

## Model Components

### Multi-Head Attention
- Scaled dot-product attention mechanism
- Multiple attention heads for learning different relationships
- Causal masking for autoregressive generation

### Transformer Block
- Pre-normalization architecture (LayerNorm before attention/FFN)
- Residual connections for gradient flow
- Position-wise feed-forward networks

### Positional Encoding
- Sinusoidal positional embeddings
- Enables the model to understand token positions

## Training Tips

1. **Start Small**: Begin with a smaller model (d_model=128, num_layers=4) for faster experimentation

2. **Data Quality**: Clean, consistent text data works better than large amounts of noisy data

3. **Learning Rate**: Use learning rate scheduling (cosine annealing is included)

4. **Regularization**: Adjust dropout based on your dataset size

5. **Monitoring**: Watch both training and validation loss to avoid overfitting

## Generation Strategies

### Temperature Sampling
- `temperature < 1.0`: More focused, deterministic outputs
- `temperature = 1.0`: Standard sampling
- `temperature > 1.0`: More random, creative outputs

### Top-k Sampling
- Only sample from the k most likely tokens
- Helps avoid generating very unlikely words
- Typical values: 20-50

## Examples

### Basic Training Example
```python
from mini_llm import MiniLLM, SimpleTokenizer
from training_script import TextDataset, train_model

# Your training data
texts = ["Your training text here..."]

# Build tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)

# Create model
model = MiniLLM(vocab_size=tokenizer.vocab_size)

# Train (see training_script.py for full example)
```

### Basic Generation Example
```python
from mini_llm import MiniLLM, SimpleTokenizer
import torch

# Load model and tokenizer
model = MiniLLM(vocab_size=1000)  # Use your vocab size
model.load_state_dict(torch.load('mini_llm_final.pt'))

# Generate text
prompt = torch.tensor([[1, 2, 3]])  # Your encoded prompt
generated = model.generate(prompt, max_new_tokens=50)
```

## Limitations

- **Simple Tokenizer**: Character-level tokenization is inefficient for large vocabularies
- **Limited Context**: Smaller context window compared to modern LLMs  
- **Training Data**: Requires substantial text data for good performance
- **Computational Resources**: Even "mini" models require decent hardware for training

## Future Improvements

- [ ] Implement BPE/SentencePiece tokenization
- [ ] Add support for different attention patterns
- [ ] Implement model parallelism for larger models
- [ ] Add more sophisticated sampling methods
- [ ] Create pre-trained models for different domains
- [ ] Add evaluation metrics and benchmarks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by "Attention Is All You Need" (Vaswani et al., 2017)
- Based on the GPT architecture (Radford et al., 2018)
- Educational resources from various deep learning courses and tutorials

## Citation

```bibtex
@software{mini_llm,
  title={Mini LLM: A Simple Language Model Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mini-llm}
}
```
# BERT and GPT Implementation

A PyTorch implementation of BERT and GPT models with configurable pre-training and fine-tuning capabilities. This project demonstrates the architectural similarities and differences between these two transformer-based models.

## Goal and Motivation

The primary goal of this project is to provide a deep, hands-on understanding of transformer-based language models, specifically BERT and GPT. By implementing these models from scratch, we aim to:

1. Explore the architectural nuances between bidirectional (BERT) and unidirectional (GPT) attention mechanisms
2. Demonstrate the flexibility of transformer architectures for various NLP tasks
3. Provide a clear, modular codebase for educational and research purposes
4. Showcase modern PyTorch optimizations and training techniques

This implementation is inspired by and builds upon the work of Andrej Karpathy, particularly his [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT) projects. While drawing inspiration from these projects, this implementation extends the concepts to include BERT and introduces additional optimizations and training scenarios.

## Key Features

- Unified implementation of BERT and GPT models
- Configurable model architecture (layers, heads, embedding dimensions)
- Support for multiple training scenarios:
  - BERT: Masked Language Modeling (MLM), Classification (SST-2, CoLA)
  - GPT: Autoregressive language modeling
- Distributed training support with PyTorch Distributed Data Parallel (DDP)
- Hardware optimizations: Flash Attention, Fused AdamW, Asynchronous data transfer
- Mixed precision training (FP32, FP16, BF16)
- Gradient accumulation for memory-efficient training
- Comprehensive logging and validation pipeline

## Implementation Details

### Model Architecture

Both BERT and GPT models share core transformer code, implemented in `src/transformer/transformer.py`. The key difference lies in their attention mechanisms:

- BERT: Bidirectional attention (no masking)
- GPT: Unidirectional attention (causal masking)

The models follow a topology similar to GPT-2, with modernizations such as:

- Pre-layer normalization
- Flash attention implementation
- Model sizes ranging from 45M to 90M parameters

### Training Framework

The training framework (`src/transformer/trainer.py`) supports various scenarios and optimizations:

- Configurable learning rate scheduling
- Checkpoint management for training continuation and fine-tuning
- Efficient dataset creation, pre-caching, and loading utilities

## Performance Results

### BERT Pre-Training (MLM)

![Validation accuracy approaches 60%!](./etc/assets/MLM_val_accuracy.png)


The BERT model achieved >60% MLM accuracy after pre-training from scratch on WikiText-103.

### SST-2 Fine-Tuning

![SST-2 Fine tuning accuracy reaches ~87%](./etc/assets/sst2_fine_tuning.png)  

The model achieved ~87% binary classification accuracy on the SST-2 dataset.

## Setup and Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare the dataset (see `dataset_preparation.md` for details) (sorry, this is stale)
4. Configure the model and training parameters in `config.ini`
5. Run pre-training: `python src/main.py`
6. For fine-tuning, use a pre-trained checkpoint: `python src/main.py --cp <checkpoint_path>`

For detailed configuration options and advanced usage, refer to the `etc/templates/config_template.ini` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

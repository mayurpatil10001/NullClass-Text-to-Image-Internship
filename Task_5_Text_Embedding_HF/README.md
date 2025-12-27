# Task 5: Text Embedding using Hugging Face

## Objective
This task demonstrates how to generate text embeddings using Hugging Face Transformers. Text embeddings are crucial for text-to-image generation as they convert text descriptions into numerical representations that models can process.

## Dataset Used
- Text descriptions (provided in the notebook)
- BERT model (bert-base-uncased)

## Libraries Used
- `transformers` - For BERT model and tokenizer
- `torch` - PyTorch for tensor operations
- `matplotlib` - For visualization
- `numpy` - For numerical operations

## Model/Library Used
- **BERT Model** (`bert-base-uncased`) from Hugging Face Transformers
- **BERT Tokenizer** for text preprocessing

## Implementation
The notebook `hf_text_embeddings.ipynb` includes:
- Loading BERT tokenizer and model
- Text tokenization with max_length=77 (common for text-to-image models)
- Generating text embeddings using BERT
- Visualizing token IDs

## Results & Metrics
- Successfully generates text embeddings
- Embedding shape: (batch_size, sequence_length, hidden_size)
- Handles multiple text descriptions
- Visualizes tokenization process

## Sample Outputs
- Text embeddings with shape information
- Token ID visualizations
- Multiple text description examples

## How to Run
1. Install required packages: `pip install transformers torch matplotlib numpy`
2. Open `hf_text_embeddings.ipynb` in Jupyter Notebook
3. Run all cells (first run will download BERT model)
4. Note: Requires internet connection for first-time model download


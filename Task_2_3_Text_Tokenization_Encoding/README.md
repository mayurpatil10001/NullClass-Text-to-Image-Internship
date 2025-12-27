# Task 2 & 3: Text Tokenization and Encoding

## Objective
This task demonstrates text tokenization and encoding using BERT tokenizer from Hugging Face Transformers. It covers converting raw text into token IDs that can be used for text-to-image generation models.

## Dataset Used
- Text samples (provided in the notebook)
- BERT tokenizer vocabulary (bert-base-uncased)

## Libraries Used
- `transformers` - For BERT tokenizer
- `matplotlib` - For visualization
- `numpy` - For numerical operations
- `torch` - PyTorch for tensor operations

## Model/Library Used
- **BERT Tokenizer** (`bert-base-uncased`) from Hugging Face Transformers

## Implementation
The notebook `tokenizer_encoding.ipynb` includes:
- Initialization of BERT tokenizer
- Text tokenization (converting text to tokens)
- Token encoding (converting tokens to token IDs)
- Visualization of token IDs

## Results & Metrics
- Successfully tokenizes text into subword tokens
- Converts tokens to numerical token IDs
- Visualizes token ID distributions
- Handles multiple text samples

## Sample Outputs
- Token sequences for input texts
- Token ID visualizations as bar charts
- Multiple examples demonstrating tokenization process

## How to Run
1. Install required packages: `pip install transformers torch matplotlib numpy`
2. Open `tokenizer_encoding.ipynb` in Jupyter Notebook
3. Run all cells to see tokenization examples


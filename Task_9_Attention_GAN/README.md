# Task 9: Attention GAN

## Objective
This task implements a Generative Adversarial Network (GAN) with self-attention mechanism. The attention mechanism helps the model focus on important features, improving image generation quality.

## Dataset Used
- Synthetic data generation (no external dataset required)
- Conditional labels: 'square' and 'circle'

## Libraries Used
- `torch` - PyTorch for deep learning
- `torch.nn` - Neural network modules
- `matplotlib` - For visualization
- `numpy` - For numerical operations

## Model Used
- **Generator with Self-Attention**: Neural network with attention mechanism
- **Discriminator**: Standard discriminator network
- **Self-Attention Module**: Query-Key-Value attention mechanism
- **Architecture**:
  - Generator: Linear layers + Self-Attention (102 → 128 → 784)
  - Discriminator: Linear layers (786 → 128 → 1)

## Implementation
The notebook `attention_gan.ipynb` includes:
- Self-attention module implementation
- Generator with attention mechanism
- Discriminator network
- Training loop with adversarial loss
- Loss visualization
- Sample generation

## Results & Metrics
- **Training Losses**: Generator and Discriminator loss curves
- **Generated Samples**: Visual outputs with attention mechanism
- **Model Performance**: Improved generation quality through attention
- **Accuracy**: Training convergence demonstrated through loss reduction

## Sample Outputs
- Training loss curves (Generator vs Discriminator)
- Generated shape images (squares and circles) with attention
- Comparison showing attention mechanism benefits

## How to Run
1. Install required packages: `pip install torch matplotlib numpy`
2. Open `attention_gan.ipynb` in Jupyter Notebook
3. Run all cells
4. Training will take some time (100 epochs)
5. Generated images will be displayed

## Key Features
- Self-attention mechanism for better feature focus
- Conditional generation based on labels
- Improved image quality compared to basic GAN


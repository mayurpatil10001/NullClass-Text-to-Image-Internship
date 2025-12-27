# Task 7: Conditional GAN for Basic Shapes

## Objective
This task implements a Conditional Generative Adversarial Network (CGAN) to generate basic shapes (squares and circles) based on conditional labels. This demonstrates the fundamental concept of conditional image generation.

## Dataset Used
- Synthetic data generation (no external dataset required)
- Conditional labels: 'square' and 'circle'

## Libraries Used
- `torch` - PyTorch for deep learning
- `torch.nn` - Neural network modules
- `matplotlib` - For visualization
- `numpy` - For numerical operations

## Model Used
- **Generator**: Neural network that generates 28x28 images from noise and labels
- **Discriminator**: Neural network that classifies real vs fake images with labels
- **Architecture**: 
  - Generator: Linear layers (102 → 128 → 784)
  - Discriminator: Linear layers (786 → 128 → 1)

## Implementation
The notebook `cgan_shapes.ipynb` includes:
- Generator and Discriminator model definitions
- Training loop with adversarial loss
- Loss visualization
- Sample generation
- Model saving functionality

## Results & Metrics
- **Training Losses**: Generator and Discriminator loss curves
- **Generated Samples**: Visual outputs for square and circle shapes
- **Model Accuracy**: Training convergence demonstrated through loss reduction
- **Model Files**: Saved in `saved_model/` directory
  - `generator.pth` - Generator weights
  - `discriminator.pth` - Discriminator weights
  - `training_state.pth` - Training state and losses

## Sample Outputs
- Training loss curves (Generator vs Discriminator)
- Generated shape images (squares and circles)
- Model checkpoints saved for future use

## How to Run
1. Install required packages: `pip install torch matplotlib numpy`
2. Open `cgan_shapes.ipynb` in Jupyter Notebook
3. Run all cells
4. Training will take some time (100 epochs)
5. Generated images and saved models will be in the output

## Model Weights
Model weights are saved in the `saved_model/` directory. If files are large, upload to Google Drive and add link here.


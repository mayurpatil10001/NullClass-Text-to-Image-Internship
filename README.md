# NullClass Text-to-Image Internship Project

## Project Overview

This repository contains all tasks completed during the Text-to-Image Generation internship with NullClass. The project covers the complete pipeline from text preprocessing to image generation using various deep learning techniques including GANs, attention mechanisms, and diffusion models.

## Internship Duration

27-06-2025 to 27-12-2025

## Task List

1. **Task 1: Image Loading and Display** - Basic image loading and visualization
2. **Task 2 & 3: Text Tokenization and Encoding** - Text tokenization using BERT tokenizer
3. **Task 4: Text Preprocessing** - Text cleaning and preprocessing techniques
4. **Task 5: Text Embedding using Hugging Face** - Generating text embeddings with BERT
5. **Task 6: Dataset Analysis** - COCO dataset analysis and visualization
6. **Task 7: Conditional GAN for Basic Shapes** - CGAN implementation for shape generation
7. **Task 8: Fine-tuning Text-to-Image Model** - Fine-tuning Stable Diffusion model
8. **Task 9: Attention GAN** - GAN with self-attention mechanism
9. **Task 10: End-to-End Pipeline** - Complete text-to-image generation pipeline

## Technologies Used

- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **NLP Libraries**: Hugging Face Transformers, BERT
- **Image Processing**: OpenCV, PIL/Pillow, Matplotlib
- **Diffusion Models**: Hugging Face Diffusers (Stable Diffusion)
- **Data Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib
- **Development**: Jupyter Notebooks, Python

## Repository Structure

```
NullClass-Text-to-Image-Internship/
│
├── Task_1_Image_Loading/
│   ├── image_display.ipynb
│   ├── sample_image.jpg
│   └── README.md
│
├── Task_2_3_Text_Tokenization_Encoding/
│   ├── tokenizer_encoding.ipynb
│   └── README.md
│
├── Task_4_Text_Preprocessing/
│   ├── text_cleaning.ipynb
│   └── README.md
│
├── Task_5_Text_Embedding_HF/
│   ├── hf_text_embeddings.ipynb
│   └── README.md
│
├── Task_6_Dataset_Analysis/
│   ├── dataset_analysis.ipynb
│   └── README.md
│
├── Task_7_CGAN_Basic_Shapes/
│   ├── cgan_shapes.ipynb
│   ├── saved_model/
│   └── README.md
│
├── Task_8_Finetuning_Text_to_Image/
│   ├── finetune_model.ipynb
│   └── README.md
│
├── Task_9_Attention_GAN/
│   ├── attention_gan.ipynb
│   └── README.md
│
├── Task_10_End_to_End_Pipeline/
│   ├── full_pipeline.ipynb
│   ├── gui_app.py (optional)
│   └── README.md
│
├── requirements.txt
├── Internship_Report.pdf
└── README.md
```

## Google Drive Links

### Model Weights & Large Files

If model files are too large for GitHub, they are stored in Google Drive. Add your links here:

- **Task 7 Model Weights**: [Add Google Drive link here if applicable]
- **Task 8 Fine-tuned Model**: [Add Google Drive link here if applicable]
- **COCO Dataset** (if used): [Add Google Drive link here if applicable]
- **Other Large Files**: [Add Google Drive links here if applicable]

**Note**: Update the links above with your actual Google Drive shareable links. Make sure the links are set to "Anyone with the link can view".

## How to Run the Project

### Prerequisites

1. Python 3.8 or higher
2. pip package manager
3. Jupyter Notebook
4. (Optional) CUDA-enabled GPU for faster training (Tasks 7, 8, 9, 10)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/NullClass-Text-to-Image-Internship.git
cd NullClass-Text-to-Image-Internship
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### Running Individual Tasks

Each task folder contains a Jupyter notebook (`.ipynb` file) that can be run independently:

1. Navigate to the task folder
2. Open the `.ipynb` file in Jupyter Notebook
3. Run all cells sequentially
4. Follow any additional instructions in the task-specific README.md

### Task-Specific Requirements

- **Task 1**: Requires a sample image file (`sample_image.jpg`)
- **Task 6**: Requires COCO dataset (download from https://cocodataset.org/)
- **Task 8**: Requires GPU for optimal performance (can run on CPU but slower)
- **Task 7, 9, 10**: Can run on CPU but GPU recommended for faster training

## Results & Screenshots

### Task 1: Image Loading
- Successfully loads and displays images with custom styling

### Task 2 & 3: Text Tokenization
- Tokenizes text into subword tokens
- Visualizes token ID distributions

### Task 4: Text Preprocessing
- Cleans text by removing special characters
- Analyzes word frequencies

### Task 5: Text Embedding
- Generates text embeddings using BERT
- Embedding shape: (batch_size, sequence_length, hidden_size)

### Task 6: Dataset Analysis
- Analyzes COCO dataset statistics
- Visualizes caption length distributions
- Displays sample image-caption pairs

### Task 7: CGAN
- Generates basic shapes (squares and circles)
- Training loss curves
- Model accuracy: Training convergence achieved

### Task 8: Fine-tuning
- Fine-tunes Stable Diffusion model
- Generates images from text prompts

### Task 9: Attention GAN
- Improved generation quality with attention
- Training loss curves
- Generated shape samples

### Task 10: End-to-End Pipeline
- Complete text-to-image pipeline
- Multiple text prompt examples
- Generated images for various descriptions

## Evaluation Metrics

For ML/GAN tasks (Tasks 7, 9, 10):
- **Accuracy**: Training convergence demonstrated through loss reduction
- **Loss Curves**: Generator and Discriminator losses visualized
- **Generated Samples**: Visual outputs for evaluation
- **Precision/Recall**: Can be computed for classification tasks

## Key Learnings

1. Text preprocessing and tokenization techniques
2. Text embedding generation using transformer models
3. Dataset analysis and visualization
4. GAN architecture and training
5. Conditional generation with CGAN
6. Attention mechanisms in GANs
7. Fine-tuning pre-trained diffusion models
8. End-to-end pipeline development

## Challenges and Solutions

[Add your specific challenges and how you solved them]

## Future Improvements

- Integration with more advanced models (DALL-E, Midjourney techniques)
- Improved GUI application
- Support for higher resolution image generation
- Better evaluation metrics (FID, IS scores)
- Multi-modal generation capabilities

## Contact

**Name**: Mayur Patil  
**Email**: [Your Email]  
**GitHub**: [Your GitHub Profile]

## License

[Add your license information if applicable]

## Acknowledgments

- NullClass for providing the internship opportunity
- Hugging Face for transformer models and diffusers library
- COCO dataset creators
- Open source community for various libraries and tools

---

**Note**: This repository is part of the NullClass Text-to-Image Generation Internship program. All tasks have been completed as per the internship requirements.



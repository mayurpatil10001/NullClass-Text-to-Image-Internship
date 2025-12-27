# Task 6: Dataset Analysis

## Objective
This task analyzes the COCO (Common Objects in Context) dataset, which is commonly used for text-to-image generation tasks. It examines image-caption pairs, caption lengths, and dataset statistics.

## Dataset Used
- **COCO Dataset** (Common Objects in Context)
  - Images: `train2017/` directory
  - Annotations: `annotations/captions_train2017.json`

## Libraries Used
- `json` - For reading COCO annotation files
- `PIL` (Pillow) - For image handling
- `matplotlib` - For visualization
- `numpy` - For statistical analysis
- `pandas` - For data manipulation (optional)

## Implementation
The notebook `dataset_analysis.ipynb` includes:
- Loading COCO annotations JSON file
- Extracting image and caption information
- Analyzing caption length distribution
- Visualizing sample images with captions
- Computing dataset statistics

## Results & Metrics
- Number of images in dataset
- Number of captions
- Average caption length (words)
- Minimum and maximum caption lengths
- Caption length distribution histogram
- Sample image-caption pairs

## Sample Outputs
- Dataset statistics (image count, caption count, average length)
- Caption length distribution histogram
- Sample images displayed with their captions

## How to Run
1. Download COCO dataset from: https://cocodataset.org/
2. Extract images to `train2017/` directory
3. Place annotation file at `annotations/captions_train2017.json`
4. Install required packages: `pip install pillow matplotlib numpy pandas`
5. Open `dataset_analysis.ipynb` in Jupyter Notebook
6. Update paths in the notebook if needed
7. Run all cells

## Note
If COCO dataset is not available, the notebook will display instructions on how to set it up.


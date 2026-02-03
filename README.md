# COVID-19-Radiography-Dataset-ML-Project-2025
Deep learning model used to classify chest X-ray images into COVID, Normal, Lung Opacity, and Viral Pneumonia categories, based on the COVID-19 Radiography Database.

## Dataset
The project utilizes the widely recognized COVID-19 Radiography Database, which is automatically managed via kagglehub.

**Class Distribution:**

* **Normal:** 10,192 images.
* **Lung Opacity:** 6,012 images.
* **COVID:** 3,616 images.
* **Viral Pneumonia:** 1,345 images.
* **Total:** 21,165 X-ray images.

The dataset exhibits a clear class imbalance (with the Normal class being the most prevalent), which is addressed during the data processing and training phases.

## Project Pipeline

**1. Dataset Setup:** Automatic download and directory restructuring to ensure compatibility with data generators (flattening nested subfolders).

**2. Preprocessing:** Removal of segmentation masks to focus exclusively on the radiography images for classification.

**3. Data Augmentation:** Applying transformations like rotations, zooms, and flips to enhance model generalization.

**4. Model Architecture:** The notebook implements a deep neural network (e.g., ResNet50 or similar) using Transfer Learning to leverage pre-trained weights.

**5. Training & Validation:** Optimization of hyperparameters and monitoring metrics such as Accuracy and Loss.

**6. Evaluation:** Performance assessment using a Confusion Matrix and visualization of model predictions.

## Technology Used
* **Python**
* **PyTorch / Torchvision**
* **KaggleHub (for dataset management)**
* **Matplotlib & Seaborn (for data visualization)**
* **Pandas & OS (for data handling)**

## Results
The model effectively distinguishes viral pneumonia and COVID-19 from healthy lungs or other opacities. Detailed metrics and visualization of predictions (including horizontal bar charts of class probabilities) are included at the end of the notebook.

## How to Run
1. Clone the respository

2. Install dependencies: pip install kagglehub pandas matplotlib seaborn torch torchvision

3. Run the notebook


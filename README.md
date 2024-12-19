# Testing the Brain Alignment of Models with High Effective Dimensionality

## Basic Overview
This project investigates the relationship between **effective dimensionality(ED)** and **brain alignment** in the vision models using the framework Brain-Score. Previous studies have highlighted both the benefits of low and high dimensionality in deep neural networks (DNNs). 

Specifically, a recent study found a potential connection between high ED and improved alignment with brain activity. This project aims to test the findings in the recent study by expading the scope of layers and models for investigation. Additionally, we aim to offer new perspectives on model interpretability, brain-inspired AI, and the broader connection between artificial and biological intelligence.

This project is conducted as a project2 of the CS-433 ML course Fall 2024 at EPFL. collaborating with neuroAI lab in EPFL. 

## File Configuration
- `data`
  - Directory containing the datasets used for analysis.
- `jobs`
  - Directory containing shell script used for submitting jobs to cluster.
- `logs`
  - Directory containing logs obtained from izar job submission.
- `notebooks`
  - Contains jupyter notebook for extracting features from models in interest.
- `resources`
  - Contains resources for analysis
- `ed-calculation.ipynb`
    - Jupyter notebook for calculating effective dimensionality(ED) based on extracted features from the layers vision models.
- `extraction-cat.ipynb`
  - Extracts feature from image and draws feature map
- `feature-map.ipynb`
  - Creates feature map from the feature extraction
- `layer1.png`
  - Layer1 feature map created with `extraction-cat.ipynb`
- `plots.ipynb`
  - Contains code for reproducing plots in the report
- `reshape.ipynb`
  - Jupyter notebook for reshaping the xarray formed feature extraction to numpy array.
- `wrong-ed-calculation.ipynb`
  - Contains wrong ed calculation code, calculating singular separately

## Execution Requirements
1. Clone the repository

2. Install the required dependencies:
   `pip isntall -r requirements.txt`


## Team information
| name    | sciper      |   
|-------------|-------------|
| Hamza Remmal    |   310917 |
| Lina Sadgal     |   342075  |
| Ahyoung Seo    | 390238    |


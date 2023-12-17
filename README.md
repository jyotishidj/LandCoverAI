# Semantic Segmentation of Land Cover Using Hyper Spectral Images

This project focuses on the semantic segmentation of land usage, categorizing areas into five distinct classes: buildings, woodlands, water, roads, and background. The dataset used for this task can be accessed on Kaggle at: https://www.kaggle.com/datasets/adrianboguszewski/landcoverai/data. 

## Project Overview
We have developed a comprehensive pipeline for the entire process, covering model training, evaluation, model registry, and deployment. The project utilizes DVC (Data Version Control), MLflow, Docker, and Streamlit to ensure an efficient and reproducible workflow.

## Getting Started
1. Download the Data: 
Download the dataset from Kaggle and place it in a "data" folder within your working directory.

2. Create a New Conda Environment:

    conda create -p venv python==3.9 -y

3. Project Template and Dependencies:
Run the following commands to set up the project template and install the required dependencies:

    python template.py

    pip install -r requirements.txt

## Code Execution

### Training and Reproducibility
Execute the following command to train the model or reproduce the existing results using DVC:


#### Option 1
python main.py

#### Option 2 (using DVC)
dvc init
dvc repro

## Streamlit Dashboard


Execute the following command to launch the Streamlit dashboard for visualizing the segmentation results:

python segment_app.py



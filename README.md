# KNN vs. Decision Tree Classification Pipeline

A comprehensive Python-based data science project developed in Google Colab. This project demonstrates an end-to-end machine learning pipeline, focusing on a comparative analysis between a K-Nearest Neighbors (KNN) model and a Decision Tree Classifier.

## Project Overview
The goal of this project is to analyze the dataset and evaluate the classification performance of two distinct algorithms: KNN (instance-based learning) and Decision Tree (eager learning), determining which approach yields higher accuracy and better generalization for this specific data.

## Dataset & Domain Focus
The dataset used in this pipeline is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 

The analytical focus is a **Binary Classification** task aimed at predicting whether a clinical sample is:
* **M (Malignant):** Cancerous
* **B (Benign):** Non-cancerous

### Feature Attributes
The dataset includes 30 predictive features describing the characteristics of the cell nuclei present in the images, including:
* **Radius, Perimeter, and Area:** Measuring the size of the nuclei.
* **Smoothness, Compactness, and Concavity:** Quantifying the shape and structural irregularities.
* **Texture and Symmetry:** Analyzing the pixel gray-scale intensity and balance.

## Technologies & Libraries Used
* **Environment:** Google Colab
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn

## Pipeline Steps & Features

### 1. Exploratory Data Analysis (EDA)
* Data loading, structure inspection, and missing value management using Pandas.
* Statistical summary and distribution analysis of the features.

### 2. Advanced Data Visualization
* **Violin Plots:** Implemented to visualize the distribution of data across different categories and detect underlying patterns.
* **Correlation Heatmaps:** Generated a feature correlation matrix to analyze multi-collinearity and select the most impactful features for the model.

### 3. Model Training & Comparative Evaluation
* **K-Nearest Neighbors (KNN):** Implemented and tuned to classify data points based on spatial proximity and feature distance.
* **Decision Tree Classifier:** Trained to learn non-linear decision boundaries, with the tree structure visualized for model explainability.
* **Performance Comparison:** Used Confusion Matrices and evaluation metrics (Accuracy, Precision, Recall) to directly compare how both models handle the dataset.

## How to Run

1. Open the `.ipynb` notebook file in Google Colab.
2. Run the first cell to initialize the environment.
3. When prompted by `files.upload()`, upload the dataset file provided in this repository.
4. Run the remaining cells sequentially to view the data analysis and model outputs.

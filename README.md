# Classifying New Particle Formation (NPF) Events

## Project Overview
This project focuses on building a machine learning classifier to predict New Particle Formation (NPF) events based on atmospheric measurements. NPF occurs when small molecules in the atmosphere form larger particles, which eventually impact cloud formation, weather patterns, and urban air pollution levels.

The project was developed as part of the Introduction to Machine Learning (Fall 2025) course at the University of Helsinki.

## Data Source
The dataset consists of atmospheric variables measured at the Hyytiälä forestry field station in Finland, primarily from the SMEAR II mast. The variables include daily means and standard deviations of various physical and chemical measurements (e.g., CO2, H2O, NO, NOx, O3, and solar radiation) taken between sunrise and sunset.

### Key Dataset Features
- **Measurements**: 104 variables including temperature (T), condensation sink (CS), and ultraviolet radiation (UV_A, UV_B).
- **Location**: Hyytiälä SMEAR II mast station.
- **Labels**: 
  - `class4`: Multi-class target (nonevent, Ia, Ib, II).
  - `class2`: Binary target (event vs. nonevent).

## Objectives
1. **Binary Classification**: Predict whether an NPF event occurred (`event`) or not (`nonevent`).
2. **Multi-class Classification**: Distinguish between specific types of NPF events (Ia, Ib, II) and nonevents.
3. **Probabilistic Prediction**: Estimate the probability of an event occurrence to minimize perplexity.

## Methodology

### 1. Exploratory Data Analysis (EDA)
Comprehensive analysis was performed on 450 training samples to understand feature distributions and correlations between atmospheric conditions and particle formation.

### 2. Preprocessing
- **Feature Selection**: Removal of non-predictive columns such as `id` and `date`.
- **Label Engineering**: Generation of binary labels from multi-class data.
- **Scaling**: Implementation of `StandardScaler` to normalize feature ranges for model stability.

### 3. Model Selection
- **Binary Classifier**: Logistic Regression was utilized for its interpretability and robust performance on the provided feature set.
- **Multi-Class Classifier**: A One-vs-Rest (OvR) Logistic Regression strategy was employed to handle the four-class classification task.

## Performance Metrics
The model was evaluated using a local Kaggle-style validation score comprising accuracy and perplexity:

- **Binary Accuracy**: 0.87778
- **Multi-class Accuracy**: 0.63333
- **Perplexity**: 8.46367
- **Estimated Kaggle Score**: 0.50370

## Repository Structure
- `npf_classification.ipynb`: Jupyter Notebook containing data analysis and modeling.
- `train.csv`: Training dataset.
- `test.csv`: Test dataset for predictions.
- `iml2025_term_project.pdf`: Scientific background and project requirements.

## Installation and Usage
To run this project locally, ensure you have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn

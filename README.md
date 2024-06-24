# Network Intrusion Detection with Machine Learning
 
This repository contains a comprehensive project focused on detecting network intrusions using various machine learning models. The project includes data preprocessing, exploratory data analysis, feature selection, model training, and evaluation to build robust classifiers for identifying potential network threats accurately.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Optimization](#models-and-optimization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The objective of this project is to develop machine learning models capable of detecting network intrusions. The steps involved in the project are:

1. Data Loading and Initial Inspection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Selection
5. Model Training and Evaluation
6. Model Optimization
7. Performance Comparison

## Dataset

The dataset used in this project is divided into training and testing sets:

- **Train_data.csv**: Contains the training data.
- **Test_data.csv**: Contains the testing data.

Both files are located in the `Dataset` directory.

## Requirements

The project requires the following Python libraries:
- [![numpy](https://img.shields.io/badge/numpy-2.0-blue)](https://numpy.org/)
- [![pandas](https://img.shields.io/badge/pandas-2.2.2-black)](https://pandas.pydata.org/)
- [![seaborn](https://img.shields.io/badge/seaborn-0.13.2-black)](https://github.com/mwaskom/seaborn.git)
- [![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-blue)](https://matplotlib.org/)
- [![warnings](https://img.shields.io/badge/warnings-built--in-red)](https://docs.python.org/3/library/warnings.html)
- [![optuna](https://img.shields.io/badge/optuna-3.3.0-blueviolet)](https://optuna.org/)
- [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
- [![lightgbm](https://img.shields.io/badge/lightgbm-4.0.0-green)](https://lightgbm.readthedocs.io/)
- [![catboost](https://img.shields.io/badge/catboost-1.1.2-yellow)](https://catboost.ai/)
- [![xgboost](https://img.shields.io/badge/xgboost-1.7.6-brightgreen)](https://xgboost.readthedocs.io/)
- [![tabulate](https://img.shields.io/badge/tabulate-0.9.0-lightgrey)](https://github.com/astanin/python-tabulate)


## Installation

To install the required libraries, you can use the following command:

```bash
pip install numpy pandas seaborn matplotlib optuna scikit-learn lightgbm catboost xgboost tabulate
```


## Usage
To run the project, follow these steps:

Clone the repository:

```bash
git clone https://github.com/yourusername/network-intrusion-detection.git
```
## Navigate to the project directory:

```bash
cd network-intrusion-detection
```
Run the Python script:

```bash
python intrusion_detection.py
```
## Models and Optimization
The project includes the following machine learning models:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- AdaBoost
- CatBoost
- Naive Bayes
- Voting Classifier
- Support Vector Machine (SVM)
- Hyperparameter optimization is performed using Optuna for models like KNN, Decision Tree, Random Forest, and SVM.

## Results
The results of the models are summarized in a tabular format, showing both train and test accuracies. The confusion matrix of the best model (XGBoost) is also displayed to visualize the classification performance.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

# License
This project is licensed under the MIT License.
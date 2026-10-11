# ML-Kernel PCA Experiments

## Project Overview
This project contains 60 independent Python programs exploring Kernel Principal Component Analysis (Kernel PCA) with different machine learning algorithms and train-test split sizes.

## Machine Learning Algorithms
- XGBoost
- Multilayer Perceptron (MLP)
- AdaBoost
- ElasticNet

## Kernel PCA Kernels
- RBF
- Linear
- Polynomial
- Sigmoid
- Cosine

## Train-Test Split Sizes
- 80:20 (test size = 0.2)
- 60:40 (test size = 0.4)
- 40:60 (test size = 0.6)

## Project Structure
Each algorithm has 15 Python programs, covering five kernels and three train-test split sizes.

**Total: 4 algorithms × 5 kernels × 3 split sizes = 60 programs.**

## Requirements
Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

## Dataset
The experiments use CIC-IDS2017 CSV files. Place the required dataset files in the appropriate working directory before running the programs.

## Important Note
ElasticNet is a regression algorithm. If its continuous predictions are converted into class labels for classification metrics, interpret those results cautiously.

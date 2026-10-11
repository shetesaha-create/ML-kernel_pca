# 60 Kernel PCA Machine Learning Experiments

This repository contains **60 independent Python scripts** using four algorithms, five Kernel PCA kernels, and three train/test split sizes.

## Algorithms
- XGBoost Classifier
- MLP Classifier
- AdaBoost Classifier
- Elastic Net (regressor; see important note below)

## Kernel PCA kernels
- RBF
- Linear
- Polynomial
- Sigmoid
- Cosine

## Test sizes
- `0.2` (20% test)
- `0.4` (40% test)
- `0.6` (60% test)

## Repository structure

```text
KernelPCA_60_ML_Experiments/
├── XGBoost/       # 15 programs
├── MLP/           # 15 programs
├── AdaBoost/      # 15 programs
├── ElasticNet/    # 15 programs
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset files required

Place these three CIC-IDS2017 CSV files in the working directory from which you run each script, or edit the `pd.read_csv(...)` paths in the script:

- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.csv`

**Important:** Dataset files are not included in this repository. They may be large and have separate distribution terms.

## Setup on Windows / Spyder

1. Install Python packages in the same environment used by Spyder:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Open a script in Spyder.
3. Make sure the three CSV files are available at the paths used in the script.
4. Run one script at a time.

For Spyder, you may need to run the install command from the Anaconda Prompt or the terminal for the Python environment selected in Spyder.

## Experiment design

Every script independently loads and preprocesses the data, performs a stratified train/test split, fits a training-only imputer, samples up to 10,000 training rows and 2,000 test rows for computational practicality, applies Kernel PCA, trains its model, and prints a confusion matrix, accuracy, weighted precision, weighted recall, weighted F1, and classification report.

The row sampling is deliberate: Kernel PCA can require substantial memory and computation on a large intrusion-detection dataset. The sampled evaluation results are not full-test-set results.

## Important note about Elastic Net

`ElasticNet` is a **regression** algorithm, not a multiclass classifier. These scripts map its continuous predictions to the nearest encoded class solely to allow classification metrics to be calculated. These metrics should not be interpreted as a standard multiclass-classification evaluation. If the assignment requires classification, ask whether a classifier such as `SGDClassifier` or `LogisticRegression` should be used instead.

## Important note about preprocessing

The imputer is fitted on the training split only, preventing the test split from influencing the imputed feature means. This is preferable for a leakage-safe evaluation.

## GitHub

Upload the folder contents to a new GitHub repository. Do not commit large raw datasets, virtual environments, or credentials.

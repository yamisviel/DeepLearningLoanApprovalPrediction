# DeepLearningLoanApprovalPrediction

This project implements an end-to-end deep learning pipeline using PyTorch to predict loan approval status. It is designed to handle the complexities of financial data, including categorical encoding, outlier management, and feature scaling, while ensuring a robust path from training to production-ready inference.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| **data/** | Directory containing the raw and processed loan datasets. |
| **EDA.ipynb** | Jupyter Notebook containing Exploratory Data Analysis to visualize feature distributions and correlations. |
| **Deep Learning Loan Approval.py** | The core training script. Handles data preprocessing, one-hot encoding, and feature scaling. Implements a Neural Network with Early Stopping to prevent overfitting. |
| **Loan Approval Model Loading.py** | Production-ready inference script. Loads trained weights and training-time metadata to generate predictions on new, unseen data. |
| **best_model.pth** | The saved state dictionary containing the optimal weights discovered during the training phase. |
| **model_metadata.pkl** | Serialized metadata (Mean/Std/Column headers) required to ensure new data is preprocessed identically to training data. |
| **Train Val Loss Curve.png** | Visualization of the training process showing model convergence over epochs. |
| **loan_predictions.csv** | Sample output file containing loan probabilities and final approval predictions for a batch of applicants. |


To make your README.md look consistent and professional, you can use Markdown lists (using * or -) for both sections. This removes the mixed indentation and makes it very easy to scan.

Here is the code to copy-paste:
Markdown

## 🚀 Workflow Overview

* **Exploration**: The `EDA.ipynb` identifies key drivers for loan defaults (e.g., income levels and loan intent).
* **Training**: `Deep Learning Loan Approval.py` processes data and trains a multi-layer perceptron. It uses a 99th percentile cap for income to handle outliers and applies normalization logic.
* **Persistence**: The script saves the "Best" model based on validation loss, alongside a `.pkl` file containing exact scaling parameters.
* **Inference**: `Loan Approval Model Loading.py` ingests new CSV data, ensures schema consistency via `reindex`, and outputs results at a custom probability threshold.

---

## 🛠️ Requirements

To run this project, you will need **Python 3.8+** and the following libraries:

### Core Libraries
* **PyTorch**: Deep learning framework used for model architecture and training.
* **Pandas**: Data manipulation and one-hot encoding.
* **NumPy**: Numerical operations and array handling.
* **Scikit-Learn**: Used for generating classification reports and data splitting.

### Utilities
* **Joblib**: For serializing and loading the preprocessing metadata.
* **Matplotlib**: For plotting training and validation loss curves.
* **Tqdm**: For progress bars during training loops.

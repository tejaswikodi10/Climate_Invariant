Abstract

This project applies a machine learning approach to analyze a dataset related to towns in Andhra Pradesh, India. The Linear Regression model predicts the town codes based on district codes and related identifiers. Various evaluation metrics, including Mean Squared Error (MSE), R-squared (R²), F1-score, and confusion matrix analysis, are used to assess the model's performance. The project also incorporates extensive visualizations to understand feature relationships, model behavior, and prediction outcomes.

Keywords

Machine Learning, Linear Regression, Town Code Prediction, Model Evaluation Metrics, Visualization, Python

About Dataset

Source: The dataset represents administrative data from Andhra Pradesh, India, and is provided as a CSV file.
Structure:
Columns include ST, DT, SDT Code, and Town.
Non-numeric columns are encoded using Label Encoding to make them suitable for machine-learning algorithms.
Use Case: Predict the Town Code using related feature columns.

Proposed Algorithm

1. Data Preprocessing: Encode categorical variables using LabelEncoder.Visualize feature distributions and relationships with Seaborn pair plots.
2. Model Selection and Training: Select Linear Regression as the predictive model. Split the dataset into training (80%) and testing (20%)subsets.
3. Model Evaluation: Calculate MSE, R², and F1-score metrics. Analyze model behavior with visualizations, including R² comparisons, true vs. predicted value distributions, and confusion matrices.
4. Feature Importance Analysis: Use SHAP (SHapley Additive exPlanations) to interpret model predictions.
5. Result Visualization: Generate plots for accuracy, loss curves, and false positive rates across prediction bins.

Results and Performance

Metrics: Training MSE, Validation MSE, Training R², Validation R², Simulated F1-Score
Visualizations: Pair plots for feature relationships, Bar plots for MSE and R² comparisons, Histograms and scatter plots for predictions vs. true values, and SHAP summary plots for feature importance.

Reproducibility

All code is written in Python and uses popular libraries like numpy, pandas, sklearn, matplotlib, and shape.The project includes detailed comments to facilitate reproducibility.

Dependencies and Requirements

Software Requirements:

Python 3.8 or higher
TensorFlow 2.6+
Keras 2.8+
NumPy
OpenCV
Matplotlib
Scikit-learn
CUDA Toolkit (for GPU acceleration

To install dependencies, run

pip install tensorflow keras numpy opencv-python matplotlib scikit-learn nibabel

Server and Hardware Requirements

Recommended Hardware
GPU: NVIDIA RTX 3080 or higher (for optimal performance)
CPU: Intel i7 or AMD Ryzen 7 (or equivalent)
RAM: 32 GB or more
Storage: Minimum 500 GB SSD (for faster data processing)
Operating System
Linux (Ubuntu 20.04+ recommended) or Windows 10/11
Server Environment
NVIDIA CUDA drivers and cuDNN installed for GPU acceleration

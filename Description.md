𝘼𝙗𝙨𝙩𝙧𝙖𝙘𝙩

This study explores the application of linear regression to predict categorical town codes from features like ST Code, DT Code, and SDT Code in a dataset. Extensive data preprocessing, visualization, and evaluation techniques were employed to optimize performance and interpretability. Key metrics include Mean Squared Error (MSE), R-squared (R²), and F1-score. Visualization techniques such as pair plots, SHAP summaries, and correlation heatmaps provide insights into the dataset and model performance.



𝙆𝙚𝙮𝙬𝙤𝙧𝙙𝙨

Linear Regression, SHAP, Feature Encoding, Mean Squared Error, R-squared, Visualization, Predictive Analysis, Data Preprocessing, Machine Learning


𝘼𝙗𝙤𝙪𝙩 𝘿𝙖𝙩𝙖𝙨𝙚𝙩

The dataset comprises encoded categorical features and numeric identifiers:

ST Code: State Code

DT Code: District Code

SDT Code: Sub-district Code

Town Code: Categorical target variable representing town codes

The dataset was preprocessed by encoding categorical features using Label Encoding.


𝙋𝙧𝙤𝙥𝙤𝙨𝙚𝙙 𝘼𝙡𝙜𝙤𝙧𝙞𝙩𝙝𝙢

1. Data Preprocessing: Encoded non-numeric columns using LabelEncoder.
Split data into training (80%) and testing (20%) sets.

2. Model Training: Utilized LinearRegression from sklearn.
Model Evaluation:

3. Calculated metrics: MSE, R², and F1-score.
Visualized performance using R² comparisons, SHAP feature importance, and confusion matrices.

4. Visualization: Pair plots for feature interactions.
Heatmaps for correlation analysis. Training and validation metrics plotted over epochs.
5 . Interpretability: Applied SHAP for feature importance analysis.


𝙍𝙚𝙨𝙪𝙡𝙩𝙨 𝙖𝙣𝙙 𝙋𝙚𝙧𝙛𝙤𝙧𝙢𝙖𝙣𝙘𝙚

The proposed model achieved 95% classification accuracy 
metrics: Training MSE, Validation MSE, Training R², Validation R², Simulated F1-Score
Visualizations: Pair plots for feature relationships, Bar plots for MSE and R² comparisons, Histograms and scatter plots for predictions vs. true values, and SHAP summary plots for feature importance.
Performance visualizations highlight that the model generalizes well to unseen data, with minimal overfitting.


𝙍𝙚𝙥𝙧𝙤𝙙𝙪𝙘𝙞𝙗𝙞𝙡𝙞𝙩𝙮

To reproduce the results: Use the provided dataset (andhrapradesh.csv).
Follow the preprocessing, training, and evaluation pipeline.
Ensure the dependencies and hardware requirements are met.
All code is written in Python and uses popular libraries like numpy, pandas, sklearn, matplotlib, and shape.The project includes detailed comments to facilitate reproducibility.


𝘿𝙚𝙥𝙚𝙣𝙙𝙚𝙣𝙘𝙞𝙚𝙨 𝙖𝙣𝙙 𝙍𝙚𝙦𝙪𝙞𝙧𝙚𝙢𝙚𝙣𝙩𝙨

Software Requirements:

Python 3.8 or higher

TensorFlow 2.6+

Keras 2.8+

NumPy

OpenCV

Matplotlib

Scikit-learn

CUDA Toolkit (for GPU acceleration


𝙏𝙤 𝙞𝙣𝙨𝙩𝙖𝙡𝙡 𝘿𝙚𝙥𝙚𝙣𝙙𝙚𝙣𝙘𝙞𝙚𝙨, 𝙧𝙪𝙣

pip install numpy pandas matplotlib seaborn scikit-learn shap


𝙎𝙚𝙧𝙫𝙚𝙧 𝙖𝙣𝙙 𝙃𝙖𝙧𝙙𝙬𝙖𝙧𝙚 𝙍𝙚𝙦𝙪𝙞𝙧𝙚𝙢𝙚𝙣𝙩𝙨

Operating System: Windows, macOS, or Linux

RAM: Minimum 4GB (8GB recommended for larger datasets)

CPU: Dual-core processor or higher

GPU: Optional, for enhanced computation with large datasets
NVIDIA CUDA drivers and cuDNN installed for GPU acceleration

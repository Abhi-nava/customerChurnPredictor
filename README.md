# Churn Prediction with Artificial Neural Networks (ANN)

This project uses an Artificial Neural Network (ANN) to predict customer churn based on a variety of customer-related factors. The model determines the likelihood of a customer leaving a business and provides insights for targeted retention strategies.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [File Descriptions](#file-descriptions)
7. [Technologies Used](#technologies-used)
8. [Results](#results)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project is designed to help businesses predict which customers are at risk of churning (leaving). By analyzing customer data such as credit score, age, balance, and tenure, the system uses an ANN to classify customers as likely to churn or not. The streamlined preprocessing and training pipelines ensure high accuracy and robustness.

### Key Features:
- Preprocessing of categorical and numerical data for model input.
- Use of feature encoding and scaling techniques to handle various data types.
- A trained ANN model with binary classification output for churn prediction.

---

## Dataset

The dataset (`Churn_Modelling.csv`) contains information related to bank customers. Key features include:
- **CreditScore**: The customer's credit score.
- **Geography**: The customer's country of residence.
- **Gender**: The customer's gender.
- **Age**: The age of the customer.
- **Tenure**: The duration of the customer's relationship with the bank.
- **Balance**: The account balance of the customer.
- **NumOfProducts**: The number of products the customer uses.
- **HasCrCard**: Whether the customer has a credit card (binary).
- **IsActiveMember**: Whether the customer is an active member (binary).
- **Exited**: Target variable (1 if the customer churned, 0 otherwise).

Unused columns such as `RowNumber`, `CustomerId`, and `Surname` were excluded during preprocessing to streamline the data.

---

## Model Architecture

The model architecture consists of:
1. **Input Layer**:
   - Accepts preprocessed features as input.
2. **Hidden Layers**:
   - Two fully connected hidden layers:
     - Layer 1: 64 neurons with ReLU activation.
     - Layer 2: 32 neurons with ReLU activation.
3. **Output Layer**:
   - One neuron with a sigmoid activation function for binary classification.
4. **Compiler Settings**:
   - Optimizer: Adam.
   - Loss Function: Binary Crossentropy.
   - Metrics: Accuracy.

---

## Installation and Setup

### Prerequisites:
- Python 3.8 or higher.
- Virtual environment tools (e.g., conda or venv).

### Steps:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies include:
   - `TensorFlow`
   - `scikit-learn`
   - `pandas`
   - `numpy`

3. Ensure the dataset `Churn_Modelling.csv` is placed in the working directory.

4. Train the model:
   ```bash
   python train_model.py
   ```

---

## Usage

### Training the Model
The `train_model.py` script:
- Preprocesses the dataset (e.g., feature encoding and scaling).
- Splits data into training and testing sets.
- Builds and trains an ANN model to predict churn.
- Saves the trained model and preprocessing pipelines (scaler and encoders) for reuse.

### Making Predictions
Use `prediction.py` to predict churn for new customer data:
1. Modify the user input dictionary in the script with the customer's details:
   ```python
   input_data = {
       'CreditScore': 600,
       'Geography': 'France',
       'Gender': 'Male',
       'Age': 40,
       'Tenure': 3,
       'Balance': 60000,
       'NumOfProducts': 2,
       'HasCrCard': 1,
       'IsActiveMember': 1,
   }
   ```
2. Run the script:
   ```bash
   python prediction.py
   ```
   The script outputs whether the customer is likely to churn and the probability of churn.

---

## File Descriptions

- **`train_model.py`**: Handles data preprocessing, model training, and serialization of scaler/encoders.
- **`prediction.py`**: Loads the trained model and preprocessing objects to predict churn for new data.
- **`Churn_Modelling.csv`**: Raw data used for model building.
- **`model.h5`**: Saved trained ANN model weights.
- **Preprocessing files**:
  - `scaler.pkl`: StandardScaler for numerical features.
  - `label_encoder_gender.pkl`: LabelEncoder for the `Gender` feature.
  - `onehot_encoder_geo.pkl`: OneHotEncoder for the `Geography` feature.
- **`requirements.txt`**: Python dependencies for the project.

---

## Technologies Used

- **TensorFlow/Keras**: Model architecture and training.
- **scikit-learn**: Preprocessing tasks (e.g., encoding, scaling).
- **Pandas**: Data handling and transformation.
- **NumPy**: Numerical operations.
- **Matplotlib/TensorBoard**: Visualization of metrics during training.

---

## Results

The ANN model achieves excellent predictive capabilities by leveraging:
- Feature Engineering: Encoding categorical and scaling numerical values.
- Optimized Neural Network with ReLU activation and Early Stopping to prevent overfitting.
- TensorBoard Logs: Insights into model performance during training.

Example results:
- Predicted churn probability: `2.7%`
- Prediction: `The customer is not likely to churn.`

---

## Acknowledgments

This project was created to explore the practical applications of deep learning in business analytics. The dataset used was sourced from online repositories. The framework and guidelines allow for easy adaptation to other binary classification problems.
# Breast Cancer Campaign 2: Neural Networks

## Project Overview

This project applies artificial neural networks to two machine-learning tasks:

1. Breast cancer classification.
2. Medical insurance-cost regression.

The breast cancer task predicts whether a tumor is benign or malignant. The insurance task predicts an individual’s medical insurance charges using demographic and health-related features.

Two neural-network approaches are used for both tasks:

* Scikit-Learn Multi-Layer Perceptron models.
* Keras neural networks using TensorFlow.

## Datasets

### Breast Cancer Dataset

The breast cancer dataset is loaded from:

```text
data_refined.csv
```

The target column is `diagnosis`.

* `M` represents malignant tumors.
* `B` represents benign tumors.

The ID and unused empty columns are removed before model training.

### Insurance Dataset

The insurance dataset is loaded from:

```text
insurance(1).csv
```

The target column is `charges`, which represents medical insurance cost.

The predictor variables include age, sex, body mass index, number of children, smoker status, and region.

## Data Splitting

Both datasets are split into:

* 80% training data
* 10% validation data
* 10% testing data

The training data is used to fit each neural-network model. The validation data is used to monitor performance while developing the models. The test data is reserved for the final evaluation.

## Preprocessing

The following preprocessing steps are applied:

* Removal of unnecessary columns.
* Handling missing values with median values.
* Conversion of categorical insurance variables into numerical dummy variables.
* Feature scaling using `StandardScaler`.
* Scaling of insurance charges before neural-network regression.

Scaling is important because neural networks are sensitive to differences in feature ranges. Without scaling, variables with larger numerical values can dominate the training process.

## Breast Cancer Classification

Two neural-network classifiers are trained:

* `MLPClassifier` from Scikit-Learn.
* A Keras Sequential neural network.

Both models predict whether a tumor is benign or malignant.

The models are evaluated using:

* Validation accuracy
* Test accuracy
* Confusion matrix

The required minimum test accuracy for the project is 94%.

## Insurance Regression

Two neural-network regression models are trained:

* `MLPRegressor` from Scikit-Learn.
* A Keras Sequential neural network.

Both models predict medical insurance charges.

The models are evaluated using:

* Validation R² score
* Test R² score
* Mean Absolute Error

The required minimum R² score for the project is 82%.

## Output Files

The project saves all results inside an `outputs` folder:

```text
sklearn_breast_confusion_matrix.png
keras_breast_confusion_matrix.png
insurance_predictions.png
model_results.csv
```

The confusion-matrix plots show the number of correct and incorrect breast-cancer predictions. The insurance prediction plot compares actual insurance charges with predicted charges.

## Requirements

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## How to Run

Place these files in the same folder:

```text
breast_cancer_campaign_2.py
data_refined.csv
insurance(1).csv
```

Run:

```bash
python breast_cancer_campaign_2.py
```

## Conclusion

This project demonstrates how neural networks can be used for both classification and regression tasks.

For breast cancer classification, neural networks can identify patterns in tumor measurements and predict whether a sample is benign or malignant. For insurance regression, neural networks can learn relationships between personal characteristics, smoking status, body mass index, and medical costs.

The results are compared between Scikit-Learn MLP models and Keras neural networks. This comparison helps show how different neural-network implementations can perform on the same data.

The breast cancer model is for educational purposes only and should not be used as a real medical diagnostic system. Similarly, the insurance prediction model should not be used as the only basis for financial or insurance decisions.

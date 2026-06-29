import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_absolute_error
)

os.makedirs("outputs", exist_ok=True)

BREAST_PATH = "data_refined.csv"
INSURANCE_PATH = "insurance(1).csv"

breast = pd.read_csv(BREAST_PATH)
insurance = pd.read_csv(INSURANCE_PATH)

breast.columns = breast.columns.str.strip()
insurance.columns = insurance.columns.str.strip()

breast = breast.drop(columns=["id", "ID", "Unnamed: 32"], errors="ignore")

X_breast = breast.drop(columns=["diagnosis"])
y_breast = breast["diagnosis"].map({"M": 1, "B": 0})

X_breast = X_breast.apply(pd.to_numeric, errors="coerce")
X_breast = X_breast.fillna(X_breast.median())

X_train_b, X_temp_b, y_train_b, y_temp_b = train_test_split(
    X_breast,
    y_breast,
    test_size=0.20,
    random_state=42,
    stratify=y_breast
)

X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(
    X_temp_b,
    y_temp_b,
    test_size=0.50,
    random_state=42,
    stratify=y_temp_b
)

breast_scaler = StandardScaler()

X_train_b = breast_scaler.fit_transform(X_train_b)
X_val_b = breast_scaler.transform(X_val_b)
X_test_b = breast_scaler.transform(X_test_b)

sklearn_classifier = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=1500,
    random_state=42
)

sklearn_classifier.fit(X_train_b, y_train_b)

sklearn_val_pred = sklearn_classifier.predict(X_val_b)
sklearn_test_pred = sklearn_classifier.predict(X_test_b)

sklearn_val_accuracy = accuracy_score(y_val_b, sklearn_val_pred)
sklearn_test_accuracy = accuracy_score(y_test_b, sklearn_test_pred)

print("\nScikit-Learn MLP Classifier")
print("Validation Accuracy:", round(sklearn_val_accuracy, 4))
print("Test Accuracy:", round(sklearn_test_accuracy, 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test_b, sklearn_test_pred))

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test_b, sklearn_test_pred),
    display_labels=["Benign", "Malignant"]
)

display.plot()
plt.title("Scikit-Learn MLP Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/sklearn_breast_confusion_matrix.png", dpi=300)
plt.show()

tf.keras.utils.set_random_seed(42)

keras_classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_b.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

keras_classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

keras_classifier.fit(
    X_train_b,
    y_train_b,
    validation_data=(X_val_b, y_val_b),
    epochs=200,
    batch_size=32,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )
    ]
)

keras_val_prob = keras_classifier.predict(X_val_b, verbose=0)
keras_test_prob = keras_classifier.predict(X_test_b, verbose=0)

keras_val_pred = (keras_val_prob >= 0.5).astype(int).ravel()
keras_test_pred = (keras_test_prob >= 0.5).astype(int).ravel()

keras_val_accuracy = accuracy_score(y_val_b, keras_val_pred)
keras_test_accuracy = accuracy_score(y_test_b, keras_test_pred)

print("\nKeras Neural Network Classifier")
print("Validation Accuracy:", round(keras_val_accuracy, 4))
print("Test Accuracy:", round(keras_test_accuracy, 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test_b, keras_test_pred))

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test_b, keras_test_pred),
    display_labels=["Benign", "Malignant"]
)

display.plot()
plt.title("Keras Breast Cancer Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/keras_breast_confusion_matrix.png", dpi=300)
plt.show()

X_insurance = insurance.drop(columns=["charges"])
y_insurance = insurance["charges"]

X_insurance = pd.get_dummies(
    X_insurance,
    drop_first=True,
    dtype=float
)

X_insurance = X_insurance.apply(pd.to_numeric, errors="coerce")
X_insurance = X_insurance.fillna(X_insurance.median())

X_train_i, X_temp_i, y_train_i, y_temp_i = train_test_split(
    X_insurance,
    y_insurance,
    test_size=0.20,
    random_state=42
)

X_val_i, X_test_i, y_val_i, y_test_i = train_test_split(
    X_temp_i,
    y_temp_i,
    test_size=0.50,
    random_state=42
)

insurance_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_i = insurance_scaler.fit_transform(X_train_i)
X_val_i = insurance_scaler.transform(X_val_i)
X_test_i = insurance_scaler.transform(X_test_i)

y_train_i_scaled = target_scaler.fit_transform(
    y_train_i.to_numpy().reshape(-1, 1)
).ravel()

sklearn_regressor = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=3000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=40
)

sklearn_regressor.fit(X_train_i, y_train_i_scaled)

sklearn_val_pred_scaled = sklearn_regressor.predict(X_val_i)
sklearn_test_pred_scaled = sklearn_regressor.predict(X_test_i)

sklearn_val_pred_i = target_scaler.inverse_transform(
    sklearn_val_pred_scaled.reshape(-1, 1)
).ravel()

sklearn_test_pred_i = target_scaler.inverse_transform(
    sklearn_test_pred_scaled.reshape(-1, 1)
).ravel()

sklearn_val_r2 = r2_score(y_val_i, sklearn_val_pred_i)
sklearn_test_r2 = r2_score(y_test_i, sklearn_test_pred_i)
sklearn_test_mae = mean_absolute_error(y_test_i, sklearn_test_pred_i)

print("\nScikit-Learn MLP Regressor")
print("Validation R2 Score:", round(sklearn_val_r2, 4))
print("Test R2 Score:", round(sklearn_test_r2, 4))
print("Test MAE:", round(sklearn_test_mae, 2))

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

keras_regressor = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_i.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

keras_regressor.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

keras_regressor.fit(
    X_train_i,
    y_train_i_scaled,
    validation_data=(
        X_val_i,
        target_scaler.transform(
            y_val_i.to_numpy().reshape(-1, 1)
        ).ravel()
    ),
    epochs=300,
    batch_size=32,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True
        )
    ]
)

keras_val_pred_scaled_i = keras_regressor.predict(X_val_i, verbose=0)
keras_test_pred_scaled_i = keras_regressor.predict(X_test_i, verbose=0)

keras_val_pred_i = target_scaler.inverse_transform(
    keras_val_pred_scaled_i
).ravel()

keras_test_pred_i = target_scaler.inverse_transform(
    keras_test_pred_scaled_i
).ravel()

keras_val_r2 = r2_score(y_val_i, keras_val_pred_i)
keras_test_r2 = r2_score(y_test_i, keras_test_pred_i)
keras_test_mae = mean_absolute_error(y_test_i, keras_test_pred_i)

print("\nKeras Neural Network Regressor")
print("Validation R2 Score:", round(keras_val_r2, 4))
print("Test R2 Score:", round(keras_test_r2, 4))
print("Test MAE:", round(keras_test_mae, 2))

plt.figure(figsize=(8, 6))
plt.scatter(y_test_i, sklearn_test_pred_i, label="Scikit-Learn MLP")
plt.scatter(y_test_i, keras_test_pred_i, label="Keras Neural Network")

minimum_value = min(y_test_i.min(), sklearn_test_pred_i.min(), keras_test_pred_i.min())
maximum_value = max(y_test_i.max(), sklearn_test_pred_i.max(), keras_test_pred_i.max())

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("Insurance Regression Predictions")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/insurance_predictions.png", dpi=300)
plt.show()

results = pd.DataFrame({
    "Model": [
        "Scikit-Learn MLP Classifier",
        "Keras Neural Network Classifier",
        "Scikit-Learn MLP Regressor",
        "Keras Neural Network Regressor"
    ],
    "Validation_Score": [
        sklearn_val_accuracy,
        keras_val_accuracy,
        sklearn_val_r2,
        keras_val_r2
    ],
    "Test_Score": [
        sklearn_test_accuracy,
        keras_test_accuracy,
        sklearn_test_r2,
        keras_test_r2
    ]
})

results.to_csv("outputs/model_results.csv", index=False)

print("\nFinal Results")
print(results)

print("\nRequired breast cancer accuracy: at least 94%")
print("Required insurance R2 score: at least 82%")
print("All files were saved in the outputs folder.")

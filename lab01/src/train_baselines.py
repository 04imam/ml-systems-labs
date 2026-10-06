from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# 70/30 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

# Print results
print("Logistic Regression Accuracy:", round(logistic_accuracy, 4))
print("Random Forest Accuracy:", round(random_forest_accuracy, 4))

# Save results
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        round(logistic_accuracy, 4),
        round(random_forest_accuracy, 4)
    ]
})

results.to_csv(
    "lab01/results/baseline_accuracy.csv",
    index=False
)

print("Saved to lab01/results/baseline_accuracy.csv")

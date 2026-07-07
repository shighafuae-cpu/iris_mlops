"""
Simple Training Script:
    - Loads the Iris dataset
    - Splits the dataset into training and testing sets
    - Trains a Logistic Regression model
    - Saves the trained model and accuracy metrics to the artifacts directory
"""


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib
import json


def main():

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    # Save Accuracy
    acc = accuracy_score(y_test, y_pred)
    metrics = {"accuracy": float(acc)}
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Saved model to {model_path}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()



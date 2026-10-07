import os
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ------------------------------------------------
# 1. LOAD UCI SPAMBASE
# ------------------------------------------------

print("Downloading UCI Spambase dataset...")

spambase = fetch_ucirepo(id=94)

X = spambase.data.features
y = spambase.data.targets.iloc[:, 0]

print("\nDataset loaded successfully!")
print("Total emails:", len(X))
print("Number of features:", X.shape[1])

print("\nClass distribution:")
print(y.value_counts())


# ------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 3. SCALE FEATURES
# ------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------
# 4. DEFINE MODELS
# ------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}


best_model = None
best_model_name = None
best_f1 = -1


# ------------------------------------------------
# 5. TRAIN AND EVALUATE
# ------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    # Logistic Regression benefits from scaling
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"]
        )
    )

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# ------------------------------------------------
# 6. SAVE MODEL
# ------------------------------------------------

home = os.path.expanduser("~")

model_path = os.path.join(
    home,
    "spam_model.pkl"
)

scaler_path = os.path.join(
    home,
    "spam_scaler.pkl"
)

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)


# Save feature names because Gmail messages must later
# be converted into these same Spambase features.
feature_path = os.path.join(
    home,
    "spambase_features.pkl"
)

joblib.dump(list(X.columns), feature_path)


print("\n" + "=" * 55)
print("FINAL RESULT")
print("=" * 55)

print("Best model:", best_model_name)
print(f"Best Spam F1 Score: {best_f1:.4f}")

print("\nSaved:")
print(model_path)
print(scaler_path)
print(feature_path)
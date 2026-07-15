import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset/credit_card.csv")

print(df.head())
print(df.dtypes)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()   # Create encoder
        df[col] = le.fit_transform(df[col].astype(str))

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Approved", axis=1)
y = df["Approved"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save model
joblib.dump(best_model, "credit_card_model.pkl")

print("\nModel saved successfully!")
print("Best Accuracy:", best_accuracy)
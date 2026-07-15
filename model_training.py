import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset/credit_card.csv")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
categorical_cols = [
    'Gender',
    'Own_Car',
    'Own_Realty',
    'Income_Type',
    'Education',
    'Family_Status',
    'Housing_Type'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("Approved", axis=1)
y = df["Approved"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric='logloss',
            random_state=42
        )
}

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("\n" + "="*50)
    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(best_model, "credit_card_model.pkl")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("Model saved successfully!")
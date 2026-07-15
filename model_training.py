import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/credit_card.csv")

print("First 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# Remove duplicate rows
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

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

print("\nAfter Encoding:")
print(df.head())

print("\nData Types After Encoding:")
print(df.dtypes)

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

# Create model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("\nAccuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "credit_card_model.pkl")

print("\nModel saved successfully!")
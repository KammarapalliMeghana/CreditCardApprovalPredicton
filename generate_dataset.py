import pandas as pd
import random

rows = []

for _ in range(1000):
    gender = random.choice(["Male", "Female"])
    income_type = random.choice(
        ["Working", "Commercial Associate", "Pensioner"]
    )
    annual_income = random.randint(80000, 1000000)
    employment_years = random.randint(0, 30)
    education = random.choice(
        ["Secondary", "Graduate", "Post Graduate"]
    )

    # Simple approval rule
    approved = 0
    if annual_income >= 250000 and employment_years >= 2:
        approved = 1
    if education == "Post Graduate" and annual_income >= 200000:
        approved = 1

    rows.append([
        gender,
        income_type,
        annual_income,
        employment_years,
        education,
        approved
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "Gender",
        "Income_Type",
        "Annual_Income",
        "Employment_Years",
        "Education",
        "Approved"
    ]
)

df.to_csv("dataset/credit_card.csv", index=False)

print("Dataset created successfully with", len(df), "rows.")
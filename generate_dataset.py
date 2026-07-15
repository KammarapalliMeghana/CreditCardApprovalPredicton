import pandas as pd
import random

rows = []

for i in range(2000):
    gender = random.choice(["Male", "Female"])
    own_car = random.choice(["Yes", "No"])
    own_realty = random.choice(["Yes", "No"])
    annual_income = random.randint(50000, 1000000)
    income_type = random.choice(
        ["Working", "Commercial Associate", "Pensioner"]
    )
    education = random.choice(
        ["Secondary", "Graduate", "Post Graduate"]
    )
    family_status = random.choice(
        ["Single", "Married", "Divorced"]
    )
    housing_type = random.choice(
        ["House", "Apartment", "Rented"]
    )
    family_members = random.randint(1, 6)
    number_of_loans = random.randint(0, 5)

    approved = 0

    if (
        annual_income >= 250000
        and number_of_loans <= 2
        and family_members <= 4
    ):
        approved = 1

    rows.append([
        gender,
        own_car,
        own_realty,
        annual_income,
        income_type,
        education,
        family_status,
        housing_type,
        family_members,
        number_of_loans,
        approved
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "Gender",
        "Own_Car",
        "Own_Realty",
        "Annual_Income",
        "Income_Type",
        "Education",
        "Family_Status",
        "Housing_Type",
        "Family_Members",
        "Number_of_Loans",
        "Approved"
    ]
)

df.to_csv(
    "dataset/credit_card.csv",
    index=False
)

print("Dataset created successfully!")
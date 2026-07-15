from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("credit_card_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    gender = int(request.form['gender'])
    income_type = int(request.form['income_type'])
    annual_income = float(request.form['annual_income'])
    employment_years = float(request.form['employment_years'])
    education = int(request.form['education'])

    features = np.array([[
        gender,
        income_type,
        annual_income,
        employment_years,
        education
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Credit Card Approved"
    else:
        result = "Credit Card Rejected"

    return render_template('result.html',
                           prediction=result)


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
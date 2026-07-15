from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(
    "credit_card_model.pkl"
)


@app.route('/')
def home():
    return render_template(
        'index.html'
    )


@app.route('/predict', methods=['POST'])
def predict():

    features = np.array([[
        int(request.form['gender']),
        int(request.form['own_car']),
        int(request.form['own_realty']),
        float(request.form['annual_income']),
        int(request.form['income_type']),
        int(request.form['education']),
        int(request.form['family_status']),
        int(request.form['housing_type']),
        int(request.form['family_members']),
        int(request.form['number_of_loans'])
    ]])

    prediction = model.predict(
        features
    )

    if prediction[0] == 1:
        result = "Credit Card Approved"
    else:
        result = "Credit Card Rejected"

    return render_template(
        "result.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)
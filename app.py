from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('cardio_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        age = int(request.form['age'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # Create input array
        input_data = np.array([[age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

        # Predict using loaded model
        prediction = model.predict(input_data)

        result = "High Risk ⚠️" if prediction[0] == 1 else "Low Risk ✅"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('cardio_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Must match your file name

@app.route('/predict', methods=['POST'])
def predict():
    # (Form processing code here...)
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


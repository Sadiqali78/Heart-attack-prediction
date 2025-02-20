from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the pre-trained heart attack prediction model
filename = 'random_forest_heart_model.sav'  # Change to your heart attack model file
model = joblib.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        age = float(request.form['age'])
        cp = float(request.form['cp'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        ca = float(request.form['ca'])

        # Format the input data into a numpy array
        input_data = np.array([[age, cp, thalach, exang, oldpeak, ca]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Map prediction to result message
        result = "High Risk of Heart Attack" if prediction[0] == 1 else "Low Risk of Heart Attack"
        print(result)

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


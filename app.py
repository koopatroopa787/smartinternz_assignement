from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('logistic_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['red_blood_cells']),
        float(request.form['pus_cell']),
        float(request.form['blood_glucose_random']),
        float(request.form['blood_urea']),
        float(request.form['pedal_edema']),
        float(request.form['anemia']),
        float(request.form['diabetes_mellitus']),
        float(request.form['coronary_artery_disease'])
    ]
    # Convert to numpy array and reshape for prediction
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    output = 'CKD' if prediction[0] == 1 else 'Not CKD'

    return render_template('index.html', prediction=output)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask
from flask import Flask, render_template, request,jsonify
from joblib.parallel import method

from pyexpat.errors import messages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import os
app = Flask(__name__)


# Load the model and scaler
model = joblib.load('world_model/population_model.pkl')

scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from the form
            population_2020 = float(request.form['population_2020'])
            population_2015 = float(request.form['population_2015'])
            population_2010 = float(request.form['population_2010'])
            area = float(request.form['area'])
            density = float(request.form['density'])
            growth_rate = float(request.form['growth_rate'])

            # Prepare the input for the model
            input_data = np.array([[population_2020, population_2015, population_2010, area, density, growth_rate]])

            # Make prediction
            prediction = model.predict(input_data)

            # Send the prediction result back to the template
            success_message = f"Prediction successful! The predicted population is {prediction[0]:,.0f}."
            #return render_template('index.html', prediction=success_message)
            return jsonify({'prediction': success_message, 'status': 'success'})

        except Exception as e:
            # Handle any errors and display the error message
            error_message="Prediction not done due to an error:"+ str(e)
            print("Error: ",error_message)
            #return render_template('index.html', prediction=error_message,status='error')
            return jsonify({'prediction': error_message, 'status': 'error'})
if __name__ == '__main__':
    app.run()

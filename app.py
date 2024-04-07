from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
app = Flask(__name__)

# Load TensorFlow model
model = tf.keras.models.load_model('my_model.h5')

# Column names and species
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Define a function to preprocess input data
def preprocess_input(data):
    return pd.DataFrame(data, index=[0], columns=CSV_COLUMN_NAMES)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Preprocess input data
    input_data = preprocess_input([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    predictions = model.predict(input_data)

    for pred in predictions:
        class_id = np.argmax(pred)
        probability = pred[class_id]
        predicted_class = 'Prediction is "{}" with probability {:.1f}%'.format(SPECIES[class_id], 100 * probability)

    response = {
    'prediction': predicted_class,
    'flower_name': SPECIES[class_id]
    }
    
    return jsonify(response)

@app.route('/flower-types')
def flower_types():
    return render_template('flower_types.html')

@app.route('/flower-types/Setosa')
def Setosa():
    return render_template('flower_types/Setosa.html')

@app.route('/flower-types/Versicolor')
def Versicolor():
    return render_template('flower_types/Versicolor.html')


@app.route('/flower-types/Virginica')
def Virginica():
    return render_template('flower_types/Virginica.html')


if __name__ == '__main__':
    app.run(debug=False,port='0.0.0.0')
    

import pickle
from flask import Flask, request, jsonify,app,render_template,url_for
import numpy as np
import pandas as pd



app = Flask(__name__)
# Load the trained model
regression_model = pickle.load(open("linear_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict():
    data = request.json['data']
    print(data)

    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = regression_model.predict(new_data)
    print(prediction[0])
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
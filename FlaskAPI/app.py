"""
https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3pHVEVEYVlONkFzUUwxUU9ObEdBRG4tcUxld3xBQ3Jtc0tuVzVKNDd5Q0VfYnd6MkhjS2tVa0JROU9IWFBiSndBdFFtczFoaksxRlExNkRtRkEyZVFZRDc2c1NpblhjUlhibjhDREdVeEtHUFhpM1hYQkF1M3VnOHhYNDNaYmprM01GZ01DdXV3N2IwZ0UxcDNKNA&q=https%3A%2F%2Ftowardsdatascience.com%2Fproductionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

"""
import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle



def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
    # stub input features
    request_json = request.get_json()
    x = request_json['input']
    #print(x)
    x_in = np.array(x).reshape(1,-1)
    # load model
    model = load_models()
    prediction = model.predict(x_in)[0]
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)
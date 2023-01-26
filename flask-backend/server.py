from flask import Flask, jsonify, request
from flask_cors import CORS
from models.BRFSS2015_lr_prediction import predict_hd
from models.lung_prediction import predict_lc, format_data
import json

import jwt
import datetime

app = Flask(__name__)
CORS(app)

SECRET_KEY = 'secret_key' # replace this with a strong secret key


def generate_token(email):
    payload = {
        'email': email,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    print(token)
    return token


@app.route('/heart-diseases', methods=['POST'])
def predict():
    data = request.get_json()
    data = {key: float(value) for key, value in data.items()}
    print(data)
    results = predict_hd(data)
    return jsonify(results)


@app.route('/lung-cancer', methods=['POST'])
def predict_lung():
    data = request.get_json()
    data = {key: float(value) for key, value in data.items()}
    print(data)
    results = predict_lc(format_data(data))
    return jsonify(results)


# Add a new route for handling login requests
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # check email and password against database or some other authentication method
    if email == 'valid@email.com' and password == 'validpassword':
        token = generate_token(email)
        return jsonify({'token': token})
    else:
        return jsonify({'error': 'Invalid email or password'}), 401


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=443)

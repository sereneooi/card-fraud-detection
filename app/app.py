import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rfpickle_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    output = ''
    int_features = [float(x) for x in request.form.values()]
    final_features = [int_features]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        output = 'No'
    else:
        output = 'Yes'

    return render_template('index.html', prediction_text='Possibly a Fraud Cases = {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
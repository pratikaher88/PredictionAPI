from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pytest

HTTP_BAD_REQUEST=400

app= Flask(__name__)

MODEL= joblib.load('iris-rf-v1.pkl')
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']

@app.route('/predict')
def predict():

    sepal_length=request.args.get('sepal_length',default=5.8, type=float)
    sepal_width=request.args.get('sepal_width',default=3.0, type=float)
    petal_length=request.args.get('petal_length',default=3.9, type=float)
    petal_width=request.args.get('petal_width',default=1.2, type=float)

    if(sepal_length is None or sepal_width is None or petal_length is None or petal_width is None):
        message = ('Record cannot be scored because of '
                   'missing or unacceptable values. '
                   'All values must be present and of type float.')
        response = jsonify(status='error',
                           error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    features=[[sepal_length,sepal_width,petal_length,petal_width]]
    probabilities= MODEL.predict_proba(features)[0]
    label_index= probabilities.argmax()
    label=MODEL_LABELS[label_index]
    class_probabilities = dict(zip(MODEL_LABELS, probabilities))

    return jsonify(status='complete',label=label,class_probabilities=class_probabilities)

if __name__ == '__main__':
    app.run(debug=True)
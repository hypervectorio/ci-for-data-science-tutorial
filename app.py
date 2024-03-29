import numpy as np
from flask import Flask
from joblib import load

app = Flask(__name__)
model = load('./pipeline.joblib')


def get_prediction(data):
    if len(np.shape(data)) == 1:
        data = np.array(data).reshape(1, -1)

    results = list(model.predict(data))
    response = {"prediction": [int(result) for result in results]}
    return response


@app.route('/')
def health_check():
    input_vector = [0, 0, 0, 0]
    return get_prediction(input_vector)


@app.route('/<a>/<b>/<c>/<d>')
def entrypoint(a, b, c, d):
    input_vector = [a, b, c, d]
    return get_prediction(input_vector)


if __name__ == '__main__':
    app.run()


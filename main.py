from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__ )
model = pickle.load(open('randomfregressor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('result.html', prediction_text='pressure {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)











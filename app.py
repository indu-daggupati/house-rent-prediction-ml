from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("rent_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    location = int(request.form['location'])

    rent = model.predict([[area, bedrooms, location]])

    return render_template('result.html', rent=round(rent[0]))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

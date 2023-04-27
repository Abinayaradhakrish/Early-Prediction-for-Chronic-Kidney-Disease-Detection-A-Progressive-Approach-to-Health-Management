from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('CKD.pkl', 'rb'))

# Define the home route
@app.route('/home')
def home():
    return render_template('home.html')

# Define the index1 route
@app.route('/index1')
def index1():
    return render_template('index1.html')

# Define the indexnew route
@app.route('/indexnew')
def indexnew():
    return render_template('indexnew.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    sen1 = request.form['sen1']
    sen2 = request.form['sen2']
    sen3 = request.form['sen3']
    sen4 = request.form['sen4']
    sen5 = request.form['sen5']
    sen6 = request.form['sen6']
    sen7 = request.form['sen7']

    # Make a prediction using the loaded model
    prediction = model.predict([[sen1, sen2, sen3, sen4, sen5, sen6, sen7]])

    # Return the predicted result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

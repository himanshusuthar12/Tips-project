import json
import pickle

from flask import Flask,request,app,render_template,jsonify
import numpy as np
# import pandas as pd

app = Flask(__name__,template_folder='templets')
## Load the model
regmodel=pickle.load(open('Tips_project.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_freature = [float(x)for x in request.form.values()]
    final_feature = [np.array(int_freature)]
    predction = regmodel.predict(final_feature)
    output = round(predction[0])
    return render_template("home.html",prediction_text="The Size of people is {}".format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = regmodel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__=="__main__":
    app.run(debug=True)
   
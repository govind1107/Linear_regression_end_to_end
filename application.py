from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application


lin_regress = pickle.load(open('models\linreg.pkl','rb'))
scaler = pickle.load(open('models\scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        hours_studied = float(request.form.get('Hours Studied'))
        prev_scores = float(request.form.get('Previous Scores'))
        extra_curr = float(request.form.get('Extracurricular Activities'))
        sleep_hours = float(request.form.get('Sleep Hours'))
        Sample_question = float(request.form.get('Sample Question Papers Practiced'))

        new_data = scaler.transform([[hours_studied,prev_scores,extra_curr,sleep_hours,Sample_question]])
        result = lin_regress.predict(new_data)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
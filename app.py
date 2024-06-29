from flask import Flask, render_template, url_for, request
import numpy as np 
import joblib as joblib
import pickle
import pandas as pd
# from tensorflow.keras.models import load_model

with open('pipeline.pkl','rb') as f:
    pipeline = pickle.load(f)
    # ann = load_model('ann.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        city = request.form['City']
        rd_spend = float(request.form['R&D'])
        adminstration = float(request.form['Adminstration'])
        marketing = float(request.form['Marketing'])
        data = np.array([city,rd_spend,adminstration,marketing])
        encoded = pipeline['encoder'].transform([[city]]).toarray()
        print(encoded)
        data = np.insert(data,0,encoded[0,0])
        data = np.insert(data,1,encoded[0,1])
        data = np.delete(data,2)
        data[2:] = pipeline['scaler'].transform(np.array(data[2:],ndmin=2))
        print(data)
        data = np.array(data,ndmin=2,dtype=np.float16)
        prediction = pipeline['RandomForest'].predict(data)
        prediction = int(prediction[0])
        return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
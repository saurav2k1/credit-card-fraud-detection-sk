from flask import Flask, render_template, request
import numpy as np
#import pickle
import joblib
app = Flask(__name__)
filename = 'credit_card_fraud_detection_model_lr.sav'
# model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
# model = joblib.load('filename.pkl')
@app.route('/')
def index():
    
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    scaled_amount = request.form['scaled_amount']
    scaled_time = request.form['scaled_time']
    v1 = request.form['v1']
    v2 = request.form['v2']
    v3 = request.form['v3']
    v4 = request.form['v4']
    v5 = request.form['v5']
    v6 = request.form['v6']
    v7 = request.form['v7']
    v8 = request.form['v8']
    v9 = request.form['v9']
    v10 = request.form['v10']
    v11 = request.form['v11']
    v12 = request.form['v12']
    v13 = request.form['v13']
    v14 = request.form['v14']
    v15 = request.form['v15']
    v16 = request.form['v16']
    v17 = request.form['v17']
    v18 = request.form['v18']
    v19 = request.form['v1']
    v20 = request.form['v20']
    v21 = request.form['v21']
    v22 = request.form['v22']
    v23 = request.form['v23']
    v24 = request.form['v24']
    v25 = request.form['v25']
    v26 = request.form['v26']
    v27 = request.form['v27']
    v28 = request.form['v28']
      
    pred = model.predict(np.array([[scaled_amount, scaled_time, v1,------, v28]]))
print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
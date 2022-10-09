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
    V308 = request.form['V308']
    V57 = request.form['V57']
    C11 = request.form['C11']
    V114 = request.form['V114']
    V283 = request.form['V283']
    V312 = request.form['V312']
    V70 = request.form['V70']
    V319 = request.form['V319']
    V19 = request.form['V19']
    V28 = request.form['V28']
    V292 = request.form['V292']
    V28.1 = request.form['V28.1']
    V82 = request.form['V82']
    V311 = request.form['V311']
    V86 = request.form['V86']
    V26 = request.form['V26']
    V300 = request.form['V300']
    C9 = request.form['C9']
    V301 = request.form['V301']
    V88 = request.form['V88']
    V284 = request.form['V284']
    V301.1 = request.form['V301.1']
    V121 = request.form['V121']
    V56 = request.form['V56']
    D15 = request.form['D15']
    V106 = request.form['V106']
    V107 = request.form['V107']
    addr2 = request.form['addr2']
    card5 = request.form['card5']
    addr1 = request.form['addr1']
    ProductCD_H = request.form['ProductCD_H']
    ProductCD_R = request.form['ProductCD_R']
    ProductCD_S = request.form['ProductCD_S']
    ProductCD_W = request.form['ProductCD_W']
    card4_discover = request.form['card4_discover']
    card4_mastercard = request.form['card4_mastercard']
    card4_visa = request.form['card4_visa']
    card6_credit = request.form['card6_credit']
    card6_debit = request.form['card6_debit']
    card6_debit or credit = request.form['card6_debit or credit']
    P_emaildomain_summary_anonymous.com = request.form['P_emaildomain_summary_anonymous.com']
    P_emaildomain_summary_aol.com = request.form['P_emaildomain_summary_aol.com']
    P_emaildomain_summary_gmail.com = request.form['P_emaildomain_summary_gmail.com']
    P_emaildomain_summary_hotmail.com = request.form['P_emaildomain_summary_hotmail.com']
    P_emaildomain_summary_yahoo.com = request.form['P_emaildomain_summary_yahoo.com']
    Transaction_Amount_bucket_low = request.form['Transaction_Amount_bucket_low']
    Transaction_Amount_bucket_medium = request.form['Transaction_Amount_bucket_medium']

      
    pred = model.predict(np.array([[V308,   V57,    C11,    V114,   V283,   V312,   V70,    V319,   V19,    V28,    V292,   V28.1,  V82,    V311,   V86,    V26,    V300,   C9, V301,   V88,    V284,   V301.1, V121,   V56,    D15,    V106,   V107,   addr2,  card5,  addr1,  ProductCD_H,    ProductCD_R,    ProductCD_S,    ProductCD_W,    card4_discover, card4_mastercard,   card4_visa, card6_credit,   card6_debit,    card6_debit or credit,  P_emaildomain_summary_anonymous.com,    P_emaildomain_summary_aol.com,  P_emaildomain_summary_gmail.com,    P_emaildomain_summary_hotmail.com,  P_emaildomain_summary_yahoo.com,    Transaction_Amount_bucket_low,  Transaction_Amount_bucket_medium]]))
    print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
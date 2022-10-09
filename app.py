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
    V28_1 = request.form['V28_1']
    V82 = request.form['V82']
    V311 = request.form['V311']
    V86 = request.form['V86']
    V26 = request.form['V26']
    V300 = request.form['V300']
    C9 = request.form['C9']
    V301 = request.form['V301']
    V88 = request.form['V88']
    V284 = request.form['V284']
    V301_1 = request.form['V301_1']
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
    card6_debit_or_credit = request.form['card6_debit_or_credit']
    P_emaildomain_summary_anonymous_com = request.form['P_emaildomain_summary_anonymous_com']
    P_emaildomain_summary_aol_com = request.form['P_emaildomain_summary_aol_com']
    P_emaildomain_summary_gmail_com = request.form['P_emaildomain_summary_gmail_com']
    P_emaildomain_summary_hotmail_com = request.form['P_emaildomain_summary_hotmail_com']
    P_emaildomain_summary_yahoo_com = request.form['P_emaildomain_summary_yahoo_com']
    Transaction_Amount_bucket_low = request.form['Transaction_Amount_bucket_low']
    Transaction_Amount_bucket_medium = request.form['Transaction_Amount_bucket_medium']


    print(f"V308 --> {V308}")
    print(f"V57 --> {V57}")
    print(f"C11 --> {C11}")
    print(f"V114 --> {V114}")
    print(f"V283 --> {V283}")
    print(f"V312 --> {V312}")
    print(f"V70 --> {V70}")
    print(f"V319 --> {V319}")
    print(f"V19 --> {V19}")
    print(f"V28 --> {V28}")
    print(f"V292 --> {V292}")
    print(f"V28_1 --> {V28_1}")
    print(f"V82 --> {V82}")
    print(f"V311 --> {V311}")
    print(f"V86 --> {V86}")
    print(f"V26 --> {V26}")
    print(f"V300 --> {V300}")
    print(f"C9 --> {C9}")
    print(f"V301 --> {V301}")
    print(f"V88 --> {V88}")
    print(f"V284 --> {V284}")
    print(f"V301_1 --> {V301_1}")
    print(f"V121 --> {V121}")
    print(f"V56 --> {V56}")
    print(f"D15 --> {D15}")
    print(f"V106 --> {V106}")
    print(f"V107 --> {V107}")
    print(f"addr2 --> {addr2}")
    print(f"card5 --> {card5}")
    print(f"addr1 --> {addr1}")
    print(f"ProductCD_H --> {ProductCD_H}")
    print(f"ProductCD_R --> {ProductCD_R}")
    print(f"ProductCD_S --> {ProductCD_S}")
    print(f"ProductCD_W --> {ProductCD_W}")
    print(f"card4_discover --> {card4_discover}")
    print(f"card4_mastercard --> {card4_mastercard}")
    print(f"card4_visa --> {card4_visa}")
    print(f"card6_credit --> {card6_credit}")
    print(f"card6_debit --> {card6_debit}")
    print(f"card6_debit_or_credit --> {card6_debit_or_credit}")
    print(f"P_emaildomain_summary_anonymous_com --> {P_emaildomain_summary_anonymous_com}")
    print(f"P_emaildomain_summary_aol_com --> {P_emaildomain_summary_aol_com}")
    print(f"P_emaildomain_summary_gmail_com --> {P_emaildomain_summary_gmail_com}")
    print(f"P_emaildomain_summary_hotmail_com --> {P_emaildomain_summary_hotmail_com}")
    print(f"P_emaildomain_summary_yahoo_com --> {P_emaildomain_summary_yahoo_com}")
    print(f"Transaction_Amount_bucket_low --> {Transaction_Amount_bucket_low}")
    print(f"Transaction_Amount_bucket_medium --> {Transaction_Amount_bucket_medium}")

    pred=model.predict(np.array([[V308,V57,C11,V114,V283,V312,V70,V319,V19,V28,V292,V28_1,V82,V311,V86,V26,V300,C9,V301,V88,V284,V301_1,V121,V56,D15,V106,V107,addr2,card5,addr1,ProductCD_H,ProductCD_R,ProductCD_S,ProductCD_W,card4_discover,card4_mastercard,card4_visa,card6_credit,card6_debit,card6_debit_or_credit,P_emaildomain_summary_anonymous_com,P_emaildomain_summary_aol_com,P_emaildomain_summary_gmail_com,P_emaildomain_summary_hotmail_com,P_emaildomain_summary_yahoo_com,Transaction_Amount_bucket_low,Transaction_Amount_bucket_medium]]))      
#    pred = model.predict(np.array([[V308,V57,C11,V114,V283,V312,V70,V319,V19,V28,V292,V28_1,  V82,    V311,   V86,    V26,    V300,   C9, V301,   V88,    V284,   V301_1, V121,   V56,    D15,    V106,   V107,   addr2,  card5,  addr1,  ProductCD_H,    ProductCD_R,    ProductCD_S,    ProductCD_W,    card4_discover, card4_mastercard,   card4_visa, card6_credit,   card6_debit,    card6_debit_or_credit,  P_emaildomain_summary_anonymous_com,    P_emaildomain_summary_aol_com,  P_emaildomain_summary_gmail_com,    P_emaildomain_summary_hotmail_com,  P_emaildomain_summary_yahoo_com,    Transaction_Amount_bucket_low,  Transaction_Amount_bucket_medium]]))
#    pred = model.predict(np.array([[1,0,0,0,0,1,0,0,2,0,2,0,2,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1]]))


    print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
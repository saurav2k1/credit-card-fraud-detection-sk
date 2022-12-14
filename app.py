from flask import Flask, render_template, request
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
app = Flask(__name__)
filename = 'credit_card_fraud_detection_model_rf_selected_depth.sav'
model = joblib.load(filename)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    TransactionDT = request.form['TransactionDT']
    TransactionAmount = request.form['TransactionAmount']    
    C1 = request.form['C1']
    C3 = request.form['C3']
    C5 = request.form['C5']
    C6 = request.form['C6']
    C7 = request.form['C7']
    C8 = request.form['C8']
    C9 = request.form['C9']
    C10 = request.form['C10']
    C11 = request.form['C11']
    C12 = request.form['C12']
    C13 = request.form['C13']
    C14 = request.form['C14']
    card1 = request.form['card1']
    card2 = request.form['card2']
    card3 = request.form['card3']
    card5 = request.form['card5']
    D1 = request.form['D1']
    D10 = request.form['D10']
    P_emaildomain = request.form['P_emaildomain']
    V85 = request.form['V85']
    V86 = request.form['V86']
    V126 = request.form['V126']
    V282 = request.form['V282']
    V306 = request.form['V306']
    V307 = request.form['V307']
    V309 = request.form['V309']
    V316 = request.form['V316']
    addr2 = request.form['addr2']

    print(f"V86 --> {V86}")
    print(f"P_emaildomain --> {P_emaildomain}")
    print(f"TransactionAmount --> {TransactionAmount}")    

    # Here I am going to transform input from user for P_emaildomain field into numeric value expected in model after onehot encoding.
    if P_emaildomain =='aol.com':
        P_emaildomain_summary_aol_com=1.0
    else:
        P_emaildomain_summary_aol_com=0.0

    print(f"P_emaildomain_summary_aol_com --> {P_emaildomain_summary_aol_com}")

    # Here I am going to transform input from user for TransactionAmount field into bucketed numeric value expected in model after onehot encoding.
    if float(TransactionAmount) < 10.0:
        Transaction_Amount_bucket ='low'
    elif float(TransactionAmount) >= 10.0 and float(TransactionAmount) <= 100.0:
        Transaction_Amount_bucket ='medium'
    else:
        Transaction_Amount_bucket ='high'

    if Transaction_Amount_bucket =='low':
        Transaction_Amount_bucket_low =1.0
    else:
        Transaction_Amount_bucket_low =0.0


    print(f"Transaction_Amount_bucket_low --> {Transaction_Amount_bucket_low}")    

    # Transaction date here is the actual date of transaction captured from user however model expects the delta which is number of seconds 
    # between input date and 20171201 (This is start date as per input dataset documentation)

    input_transaction_date=TransactionDT
    start_date='20171201'

    # convert string to date object
    d1 = datetime.strptime(input_transaction_date, "%Y%m%d")
    d2 = datetime.strptime(start_date, "%Y%m%d")

    # difference between dates in timedelta in seconds
    delta = d1 - d2
    TransactionDT=delta.days*86400
    print(f'Difference is {TransactionDT} seconds')    

    
    input_variables = pd.DataFrame([[TransactionDT,card5,card1,addr2,C12,C13,C1,C10,C3,card3,D10,C7,V86,V85,C14,C9,C5,C11,V306,D1,C6,P_emaildomain_summary_aol_com,V309,V307,Transaction_Amount_bucket_low,V126,V282,V316,C8,card2]],
                                   columns=['TransactionDT', 'card5', 'card1', 'addr2', 'C12', 'C13', 'C1', 'C10', 'C3', 'card3', 'D10', 'C7', 'V86', 'V85', 'C14', 'C9', 'C5', 'C11', 'V306', 'D1', 'C6', 'P_emaildomain_summary_aol_com', 'V309', 'V307','Transaction_Amount_bucket_low', 'V126', 'V282', 'V316', 'C8', 'card2'],
                                   dtype=float)
    pred = model.predict(input_variables)[0]

    print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
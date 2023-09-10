import pickle
from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
standard_scaler=pickle.load(open('/config/workspace/models/student_CGPA_prediction_project_scaler_model.pkl','rb'))
ridge_model=pickle.load(open('/config/workspace/models/student_CGPA_prediction_project_ridge_model.pkl','rb'))

output_csv_file='/config/workspace/records/output_data.csv'

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        English=int(request.form.get('English'))
        Hindi=int(request.form.get('Hindi'))
        Marathi=int(request.form.get('Marathi'))
        Mathematics=int(request.form.get('Mathematics'))
        Science=int(request.form.get('Science'))
        Social_Science=int(request.form.get('Social_Science'))

        new_data_scaled=standard_scaler.transform([[English,Hindi,Marathi,Mathematics,Science,Social_Science]])
        result=ridge_model.predict(new_data_scaled)
        total_marks=English + Hindi + Marathi + Mathematics + Science + Social_Science
        per=total_marks/6
        if English>35 and Hindi>35 and Marathi>35 and Mathematics>35 and Science>35 and Social_Science>35:
            final_result="Pass"
        else:
            final_result="Fail"


        # write the input and output data in to the csv file
        with open (output_csv_file,'a',newline='')as csvfile:
            writer=csv.writer(csvfile)
            # writer.writerow([['English, Hindi, Marathi, Mathematics, Science, Social_Science','CGPA']])
            writer.writerow([[English, Hindi, Marathi, Mathematics, Science, Social_Science, result[0]]])

        return render_template('result.html',result=result[0],total_marks=total_marks,per=per,final_result=final_result)
        #return {}.format(result1)

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")

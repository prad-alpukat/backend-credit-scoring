from typing import Union
from fastapi import FastAPI

# model libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

app = FastAPI()

# predict load approval
def predict_loan_approval(model,age,balance,day,duration,campaign,pdays,previous,job,marital,education,default,housing,loan,contact,month,poutcome):
    data = {
        'age':age,
        'job':job,
        'marital':marital,
        'education':education,
        'default':default,
        'balance':balance,
        'housing':housing,
        'loan':loan,
        'contact':contact,
        'day':day,
        'month':month,
        'duration':duration,
        'campaign':campaign,
        'pdays':pdays,
        'previous':previous,
        'poutcome':poutcome
    }
    data=pd.DataFrame(data,index=[0])
    list2=['age','balance','day','duration','campaign','pdays','previous']
    LE=LabelEncoder()
    sc=StandardScaler()
    data['job']=LE.fit_transform(data['job'])
    data['marital']=LE.fit_transform(data['marital'])
    data['education']=LE.fit_transform(data['education'])
    data['default']=LE.fit_transform(data['default'])
    data['housing']=LE.fit_transform(data['housing'])
    data['loan']=LE.fit_transform(data['loan'])
    data['contact']=LE.fit_transform(data['contact'])
    data['month']=LE.fit_transform(data['month'])
    data['poutcome']=LE.fit_transform(data['poutcome'])
    data[list2]=sc.fit_transform(data[list2])
    prediction = model.predict(data)
    return  prediction[0].tolist()
    
joblib_file = "model.pkl"
model = joblib.load(joblib_file)

@app.get("/")
async def read_root():
   result = predict_loan_approval( model,58,2143,5,261,1,-1,0,'management','married','tertiary','no','yes','no','unknown','may','unknown')
   return {"message": "Welcome to the API", "result": result}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
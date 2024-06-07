from typing import Union
from fastapi import FastAPI

# model libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = FastAPI()

# predict load approval
def predict_loan_approval(model, age, balance, day, duration, campaign, pdays, previous, 
                          job, marital, education, default, housing, loan, contact, month, poutcome):
    # Create a dictionary to store the data
    data_dict = {
        'age': np.array([age]),
        'balance': np.array([balance]),
        'day': np.array([day]),
        'duration': np.array([duration]),
        'campaign': np.array([campaign]),
        'pdays': np.array([pdays]),
        'previous': np.array([previous]),
        'job': np.array([job]),
        'marital': np.array([marital]),
        'education': np.array([education]),
        'default': np.array([default]),
        'housing': np.array([housing]),
        'loan': np.array([loan]),
        'contact': np.array([contact]),
        'month': np.array([month]),
        'poutcome': np.array([poutcome])
    }
    
    # List of numerical columns to scale
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    # Encode categorical features using dictionaries
    LE = LabelEncoder()
    for feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        data_dict[feature] = LE.fit_transform(data_dict[feature])
    
    # Scale numerical features using StandardScaler
    sc = StandardScaler()
    for feature in numerical_columns:
        data_dict[feature] = sc.fit_transform(data_dict[feature].reshape(-1, 1)).flatten()

    # Combine all features into a single NumPy array
    data_array = np.column_stack([data_dict[feature] for feature in data_dict.keys()])

    # Predict the loan approval
    prediction = model.predict(data_array)
    return prediction[0].tolist()
    
joblib_file = "model.pkl"
model = joblib.load(joblib_file)

@app.get("/")
async def read_root():
   result = predict_loan_approval( model,58,2143,5,261,1,-1,0,'management','married','tertiary','no','yes','no','unknown','may','unknown')
   return {"message": "Welcome to the API", "result": result}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
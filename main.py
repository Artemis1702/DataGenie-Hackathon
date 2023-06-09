from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime
import pandas as pd
import numpy as np
from DataGenie_HaCK import connect



app = FastAPI()


class DateListRequest(BaseModel):
    date_from: date
    date_to: date



@app.post("/predict")
async def predict(date_from: str, date_to: str, period: Optional[int] = 0):
    # Parse the input dates
    if period == 0:
        date_from = datetime.strptime(date_from, "%Y-%m-%d")
        date_to = datetime.strptime(date_to, "%Y-%m-%d")

    
    # Call function to determine the best time series model and make predictions
    best, mape, pred1, ind, val1 = connect(date_from, date_to, period)

    
    # Convert the data given by function to python float from np.float
    pred = [float(x) for x in pred1]
    # ind = [float(x) for x in ind1]
    val = [float(x) for x in val1]

    # Add all of the results to a dict
    result = []
    for i in range(len(pred)):
        result.append({'point_timestamp': ind[i], 'point_value': val[i], 'yhat': pred[i]})

    output = {
        "model": best,
        "mape": mape,
        "result": result 
    }

    return output
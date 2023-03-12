from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime
import pandas as pd
import numpy as np
from DataGenie_HaCK import connect



app = FastAPI()

@app.get("/{id}")
def read_root(id: int):
    return {"Hello": id}


class DateListRequest(BaseModel):
    date_from: date
    date_to: date



@app.post("/predict")
async def predict(date_from: str, date_to: str, period: Optional[int] = 0):
    # Parse the input dates
    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")


    
    # Generate the time series data based on the input dates
    index = pd.date_range(date_from, date_to, freq="D")
    data = pd.DataFrame(index=index)

    np.savetxt('C:\\Users\\tejas\\Desktop\\trial.tct', data.values)
    # Call function to determine the best time series model and make predictions
    best, mape, pred1, ind, val1 = connect(date_from, date_to)


    pred = [float(x) for x in pred1]
    # ind = [float(x) for x in ind1]
    val = [float(x) for x in val1]

    result = []
    for i in range(len(pred)):
        result.append({'point_timestamp': ind[i], 'point_value': val[i], 'yhat': pred[i]})

    output = {
        "model": best,
        "mape": mape,
        "result": result 
    }

    return output
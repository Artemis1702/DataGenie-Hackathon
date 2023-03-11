from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime
import pandas as pd
import numpy as np
import import_ipynb
from DataGenie_HaCK import selection, prediction



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
    best = selection(data)
    b_mape = prediction(best)

    output = {
        "model": best,
        "mape": b_mape,
        # "predictions": results["predictions"].tolist(),
        # "index": results["predictions"].index.format()
    }

    return output
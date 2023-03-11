from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime
import pandas as pd



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


    # do something with the dates
    
    # Generate the time series data based on the input dates
    index = pd.date_range(date_from, date_to, freq="D")
    data = pd.DataFrame(index=index)
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Define the function to call the API
def call_api(date_from, date_to, period):
    # Define the query parameters
    query_params = {
        "date_from": date_from,
        "date_to": date_to,
        "period": period
    }
    
    # Make the API request
    response = requests.post(API_URL, params=query_params)
    
    # Parse the response
    if response.status_code == 200:
        data = response.json()
        return data["model"], data["mape"], data["result"]
    else:
        st.error("Error calling API: {}".format(response.text))
        return None, None, None

# Define the Streamlit app
def run_streamlit_app():
    st.title("Time Series Prediction")
    
    # Define the input fields
    date_from = st.date_input("From date")
    date_to = st.date_input("To date")
    period = st.number_input("Period", min_value=0, max_value=None, value=0)
    
    # Call the API when the user clicks the "Predict" button
    if st.button("Predict"):
        model, mape, result = call_api(str(date_from), str(date_to), period)
        
        # Plot the results using Plotly
        if result is not None:
            df = pd.DataFrame(result)

            # Plot the line for point_value with dashed style
            df = pd.DataFrame(result)
            fig = px.line(df, x="point_timestamp", y=["point_value", "yhat"])
            st.plotly_chart(fig)
        else:
            st.error("Error: could not generate plot.")

# Run the Streamlit app
if __name__ == '__main__':
    run_streamlit_app()

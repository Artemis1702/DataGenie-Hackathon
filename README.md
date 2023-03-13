# DataGenie-Hackathon
This repository contains my code for the DataGenie hackathon conducted froom 11th March, 2023 to 13th March, 2023
## Problems Faced
* Understanding the problem statement
* Installing fbProphet
* Trying to use .ipynb files in FastAPI
* Deploying the UI
## My Approach
* I implemented a suite of four Time Series analysis models 
  * ARIMA
  * SARIMA
  * XGBoost
  * ETS
* For the selection algorithm I used a few sample datasets as training data and for each dataset I checked which Time Series model gave me the least MAPE value. I then incremented the odds of choosing that Time Series model based on the least MAPE values. In the after training completely I got the Time Series analysis model which had the most number of least MAPE values for prediction of the new dataset given. Then that dataset is also added to the training data to evaluate for the forthcoming datasets. You can enhance this algorithm by using a neural network and confusion matrix, which will give us even greater learing results. The datasets used are 
  * sample_data2 daily (Training)
  * sample_data3 daily (Training)
  * sample_data4 daily (Training)
  * sample_data5 daily (Training)
  * sample_data1 daily (Testing)
* I then used the best Time Series model decided by my selection algorithm and used it to predict for the given data. I also added plots to help understand better.
* For me the best model was XGBoost which gave very minute MAPE(consistently around 0.00034). Due to this If we look at the graphs we would not be able to see much difference.
* I then created a function called connect() which returns all the parameters required by the response body of the FastApi and takes the request body as the input parameters.
* I first tried to connect the FastApi app to the .ipynb file but after numerous attempts I decided to convert the .ipynb file to a .py file and then I could use it in the FastApi app.
* I then used streamlit to create a simple UI which has the three required fields of the FastApi request body and sends it to the FastAPI app which does the prediction and returns back the response body. I then extracted the points to be plotted from this response body in the streamlit app and plotted it using plotly.express.
## Outputs
* FastAPI outputs
  * The logic behind my response is that, as we require both y hat and point_value (inferred from the example given) we can only take values in the given dataset and check how well our Time Series model is doing in predicting the data given the actual data. As you can see the values of yhat and point_value are really close proving that, the Time Series model we used is really effective and has very less MAPE
<img width="907" alt="datagenie post 1" src="https://user-images.githubusercontent.com/76508539/224571195-946b00a1-b6fb-4551-bb23-328132074b5b.png">
<img width="904" alt="datagenie post 2" src="https://user-images.githubusercontent.com/76508539/224571193-480f37cb-3ee3-4a87-8409-c9aa538bbbc8.png">

<img width="906" alt="datagenie post 3" src="https://user-images.githubusercontent.com/76508539/224571194-bac8a6c0-9873-47eb-bf26-3561154af7a0.png">
<img width="908" alt="datagenie post 4" src="https://user-images.githubusercontent.com/76508539/224571163-2de9b63c-f3b7-4da9-970e-ffb738af5856.png">

* Streamlit output
  * I tried deploying my ui in streamlit.io by uploading it in their cloud, but I was unable to do it so as it said that plotly.express wasd not a module present there. So I have left a video demonstrating the working of my ui running on my computer.
<img width="909" alt="datagenie ui" src="https://user-images.githubusercontent.com/76508539/224571447-f78317e9-eba1-454d-874b-8ed5d47be7a3.mp4">


  
  

import streamlit as st  #imports streamlit library helps to create web applications for data science project
import pandas as pd  #pnadas helps for data manipulation and analysis for handling data in dataframe format 
import matplotlib.pyplot as plt
import seaborn as sns #seaborn is a statistical data visualization
from sklearn.preprocessing import LabelEncoder # labelencoder converts categorical lables into numerical values
from sklearn.model_selection import train_test_split  # this splits the dataset into training and testing
from sklearn.linear_model import LinearRegression  #linear regression algorithms to model the relationship between input features and output labels.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  #his imports several metrics from Scikit-learn that are used to evaluate the performance of regression models: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
import math  #This imports the math module, which provides access to mathematical functions. It will be used here specifically for calculating the square root.

st.title('Tip Prediction')  # this sets the title of streamlit application it will displayed at the top 

data = pd.read_csv("tips.csv")  # this will reads a CSV file 

data = data.drop(["smoker", "Payer Name", "CC Number", "Payment ID"], axis=1) # this will drop the not needed columns

label_encoder = LabelEncoder()  #will be used to convert categorical variables into numerical format.
data['sex_encoded'] = label_encoder.fit_transform(data['sex'])  # Female 0, male 1
data['time_encoded'] = label_encoder.fit_transform(data['time'])
data['day_encoded'] = label_encoder.fit_transform(data['day'])


x = data[['total_bill', 'sex_encoded', 'day_encoded', 'time_encoded', 'size', 'price_per_person']] # this creates a new dataframe "x" containing only features that will be used for making predictions
y = data[['tip']] # this is the new dataframe "y" which contains the targeted variable that we want to predict "tip"

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # splits into training and testing 80% is for training and 20% for testing 

model = LinearRegression() # This initializes a linear regression model instance called model.

model.fit(x_train, y_train) #This fits (trains) the linear regression model using the training data (x_train and y_train).


st.write("### Input Data for Tip Prediction") #This writes a subheading in the Streamlit app indicating that users can input data for tip prediction.


total_bill = st.number_input("Total Bill ($)", min_value=0.0, value=10.0)  #This creates a number input widget in Streamlit where users can enter the total bill amount. It has a minimum value of 0.0 and defaults to 10.0.

#These lines create various input widgets (select boxes and sliders) allowing users to specify their sex, day of the week, time of day, party size, and price per person.
sex = st.selectbox("Sex", options=['Female', 'Male'])
day = st.selectbox("Day of the Week", options=['Thur', 'Fri', 'Sat', 'Sun'])
time = st.selectbox("Time of Day", options=['Lunch', 'Dinner'])
size = st.slider("Party Size", min_value=1, max_value=10, value=2)
price_per_person = st.number_input("Price Per Person ($)", min_value=0.0, value=5.0)


#These lines encode user inputs into numerical format. For sex, it uses a conditional statement; for day and time, it applies LabelEncoder again to get their respective encoded values.
sex_encoded = 0 if sex == 'Female' else 1
day_encoded = label_encoder.fit_transform([day])[0]
time_encoded = label_encoder.fit_transform([time])[0]


#Here, a new DataFrame called input_data is created with user inputs structured in a format suitable for prediction by including all relevant features.
input_data = pd.DataFrame([[total_bill, sex_encoded, day_encoded, time_encoded, size, price_per_person]],
                          columns=['total_bill', 'sex_encoded', 'day_encoded', 'time_encoded', 'size', 'price_per_person'])

#[0][0] stands for 2D array
predicted_tip = model.predict(input_data)[0][0]  # this uses the trained model to predict tip based on user input contained in "input_data", this is to get a single numerical value from the prediction output


st.write(f"### Predicted Tip: ${predicted_tip:.2f}")

st.write("### Model Performance Metrics")  # this writes another subheading indicating that model performance metrics will be displayed next
y_pred = model.predict(x_test) #Here, predictions are made on the test set (x_test) using the trained model and stored in y_pred.

#this lines calculate various performance metrices using (y_test) and predicted labels (y_pred).

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)


st.write(f"Mean Absolute Error: {mae:.2f}")  #this outputs MAE to the Streamlit app formatted to two decimal places
# st.write(f"Mean Squared Error: {mse:.2f}")
# st.write(f"Root Mean Squared Error: {rmse:.2f}")
# st.write(f"R-squared: {r2:.2f}")

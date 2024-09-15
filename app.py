import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

st.title('Tip Prediction')

data = pd.read_csv("tips.csv")

data = data.drop(["smoker", "Payer Name", "CC Number", "Payment ID"], axis=1)

label_encoder = LabelEncoder()
data['sex_encoded'] = label_encoder.fit_transform(data['sex'])  # Female 0, male 1
data['time_encoded'] = label_encoder.fit_transform(data['time'])
data['day_encoded'] = label_encoder.fit_transform(data['day'])


x = data[['total_bill', 'sex_encoded', 'day_encoded', 'time_encoded', 'size', 'price_per_person']]
y = data[['tip']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)


st.write("### Input Data for Tip Prediction")


total_bill = st.number_input("Total Bill ($)", min_value=0.0, value=10.0)
sex = st.selectbox("Sex", options=['Female', 'Male'])
day = st.selectbox("Day of the Week", options=['Thur', 'Fri', 'Sat', 'Sun'])
time = st.selectbox("Time of Day", options=['Lunch', 'Dinner'])
size = st.slider("Party Size", min_value=1, max_value=10, value=2)
price_per_person = st.number_input("Price Per Person ($)", min_value=0.0, value=5.0)


sex_encoded = 0 if sex == 'Female' else 1
day_encoded = label_encoder.fit_transform([day])[0]
time_encoded = label_encoder.fit_transform([time])[0]


input_data = pd.DataFrame([[total_bill, sex_encoded, day_encoded, time_encoded, size, price_per_person]],
                          columns=['total_bill', 'sex_encoded', 'day_encoded', 'time_encoded', 'size', 'price_per_person'])


predicted_tip = model.predict(input_data)[0][0]


st.write(f"### Predicted Tip: ${predicted_tip:.2f}")

st.write("### Model Performance Metrics")
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Absolute Error: {mae:.2f}")
# st.write(f"Mean Squared Error: {mse:.2f}")
# st.write(f"Root Mean Squared Error: {rmse:.2f}")
# st.write(f"R-squared: {r2:.2f}")

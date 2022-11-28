import numpy as np
import xgboost

import pandas as pd
import pickle

import streamlit as st

# Toy data for model is taken from sklearn 'diabetes' dataset (excluded s1, s2, s4-s6 features)
# Final predictors are: ['age', 'sex', 'bmi', 'bp', 's3'] where 'bmi' - body mass index, 'bp' - average blood pressure, 's3' - blood serum measurement
# Model trained: 
#                XGBRegressor(
#                        learning_rate=0.1, n_estimators=140, max_depth=3,
#                        min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                        objective= 'reg:squarederror', nthread=4, scale_pos_weight=1, seed=27
#                    )

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
        
model = pickle.load(open('models/xgb_v1.pkl', 'rb'))

st.header('Input data to make a forecast')

gender_dict = {"Male": 1, "Female": 2}
#age = st.number_input('Input age', 1, 120)
age = st.slider('Age', 1, 100, 1,)
sex = gender_dict[st.radio('Pick your gender', ['Male', 'Female'])]
bmi = st.slider(label='Body Mass Index (BMI)', min_value=10.0, max_value=40.0, value=25.0, step=0.1)
bp = st.slider(label='Average Blood Pressure (BP)', min_value=80.0, max_value=170.0, value=100.0, step=1.0)
s3 = st.slider(label='Serum in blood', min_value=80.0, max_value=200.0, value=110.0, step=1.0)

# xgboost needs to DataFrame convertion
features_names = ['age', 'sex', 'bmi', 'bp', 's3']
predict_features = pd.DataFrame(np.array([age, sex, bmi, bp, s3]).reshape(1, -1), columns=features_names)

if st.button('PREDICT'):
    prediction = model.predict(predict_features)
    st.text(body='Predicted Response of interest: {:.1f}'.format(prediction[0]))

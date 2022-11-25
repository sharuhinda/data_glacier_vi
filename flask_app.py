import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, render_template

app = Flask(__name__)

"""
Toy data for model is taken from sklearn 'diabetes' dataset (excluded s1, s2, s4-s6 features)
Final predictors are: ['age', 'sex', 'bmi', 'bp', 's3'] where 'bmi' - body mass index, 'bp' - average blood pressure, 's3' - blood serum measurement
Model trained: 
                XGBRegressor(
                        learning_rate=0.1, n_estimators=140, max_depth=3,
                        min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                        objective= 'reg:squarederror', nthread=4, scale_pos_weight=1, seed=27
                    )

"""

model = pickle.load(open('week5/xgb_v1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('week5/templates/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    # xgboost needs to DataFrame convertion
    features_names = ['age', 'sex', 'bmi', 'bp', 's3']
    predict_features = pd.DataFrame(np.array([[float(x) for x in request.form.values()]]), columns=features_names)
    
    prediction = model.predict(predict_features)
    return render_template('week5/templates/index.html', prediction_text='Predicted Response of interest: {:.1f}'.format(prediction[0]))



if __name__ == "__main__":
    app.run(port=5000, debug=True)
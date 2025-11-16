import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib
from scipy.stats import randint

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs,cat_attribs):
    num_pipepline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline= Pipeline([
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('numerical',num_pipepline,num_attribs),
        ('categorical',cat_pipeline,cat_attribs)
    ])

    return full_pipeline

def model_predict():

    print('Loading Pre-trained Model and pipeline...')
    pipeline=joblib.load('pipeline.pkl')
    model=joblib.load('model.pkl')

    print('input Dataset Loading...')
    if not os.path.exists('input.csv'):
        input_data=pd.read_csv('test_data.csv').drop('G3',axis=1)
    else:
        input_data=pd.read_csv('input.csv')

    print('Preprocessing Input...')
    transformed_input=pipeline.transform(input_data)

    print('Model is Predicting...')
    input_data['predicted_G3']=model.predict(transformed_input)

    input_data.to_csv('predicted-performance.csv', index=False)

    print('Predictions saved to predicted-performance.csv')


if not os.path.exists(MODEL_FILE):
    print('Dataset Loading...')

    student_performance = pd.read_csv('student-performance.csv')

    X=student_performance.drop('G3',axis=1)
    y=student_performance['G3'].copy()

    print('Dataset Splitting...')
    X_train,X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

    pd.concat([X_test,y_test],axis=1).to_csv('test_data.csv', index=False)
    
    num_attribs = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print('Pipeline Building...')
    pipeline=build_pipeline(num_attribs,cat_attribs)

    print('Traning data Preprocessing...')
    transformed_data = pipeline.fit_transform(X_train)

    print('Model loading and fine tuning using GridSearchCV')
    rnd_forest_reg = RandomForestRegressor(random_state=42)

    
    grid_params={
    'max_depth': [15,20,25],
    'min_samples_split': [3,4,5],
    'n_estimators': [250,265,280]
}

    tuned_model=GridSearchCV(
        rnd_forest_reg,
        grid_params,
        cv=10,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    tuned_model.fit(transformed_data,y_train)

    print('Dumpinp Model and Pipeline using JobLib, no need to train again and again')
    joblib.dump(tuned_model,'model.pkl')
    joblib.dump(pipeline,'pipeline.pkl')

    model_predict()

else:

    model_predict()

    

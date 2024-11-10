# Load your trained model
import pickle
import gradio as gr
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import Any

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.metrics import r2_score,accuracy_score, f1_score, precision_score, recall_score


api_router = APIRouter()

curr_path = str(Path(__file__).parent)

title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"


# FastAPI object
app = FastAPI()

root_router = APIRouter()

import prometheus_client as prom

# Load the model from the .pkl file
with open('xgboost-model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

acc_metric = prom.Gauge('patient_survival_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('patient_survival_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('patient_survival_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('patient_survival_recall_score', 'Recall score for few random 100 test samples')
r2_metric = prom.Gauge('patient_survival_r2_score', 'R2 score for random 100 test samples')

# LOAD TEST DATA
test_data = pd.read_csv(curr_path + "/test_dataset.csv")

# Function for updating metrics
def update_metrics():
    test = test_data.sample(10)
    test_feat = test.drop('Death_event', axis=1)
    test_cnt = test['Death_event'].values
    test_pred = loaded_model.predict(test_feat)
    
    r2 = r2_score(test_cnt, test_pred)
    r2_metric.set(r2)
    
    acc = accuracy_score(test_cnt, test_pred)
    acc_metric.set(acc)

    f1 = f1_score(test_cnt, test_pred)
    f1_metric.set(f1)

    precision = precision_score(test_cnt, test_pred)
    precision_metric.set(precision)

    recall = recall_score(test_cnt, test_pred)
    recall_metric.set(recall)

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

    
def predict_death_event(age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time):
    age=float(age)
    anaemia=int(anaemia)
    creatinine_phosphokinase=int(creatinine_phosphokinase)
    diabetes=int(diabetes)
    ejection_fraction=int(ejection_fraction)
    high_blood_pressure=int(high_blood_pressure)
    platelets=float(platelets)
    serum_creatinine=float(serum_creatinine)
    serum_sodium=int(serum_sodium)
    sex=int(sex)
    smoking=int(smoking)
    time=int(time)

    input = [age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]
    input_df = pd.DataFrame([input],columns = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time'])
    pred_results = loaded_model.predict(input_df)
    return pred_results[0]

age = gr.Slider(0, 100, label="age", value=70, info="Choose age between 0 and 100")
anaemia = gr.Radio(["0", "1"], label="anaemia",value="0", info="Does patient have anaemia ->  0-False, 1-True")
creatinine_phosphokinase = gr.Slider(0, 1000, label="creatinine_phosphokinase",value=161, info="Choose creatinine_phosphokinase between 0 and 1000")
diabetes = gr.Radio(["0", "1"], label="diabetes", value="0",info="Does patient have diabetes ->  0-False, 1-True")
ejection_fraction = gr.Slider(0, 100, label="ejection_fraction",value=25, info="Choose ejection_fraction between 0 and 100")
high_blood_pressure = gr.Radio(["0", "1"], label="high_blood_pressure",value="0", info="Does patient have High BP ->  0-False, 1-True")
platelets = gr.Slider(25000, 850000, label="platelets",value=244000, info="Choose platelets between 25000 and 850000")
serum_creatinine = gr.Slider(0, 10, label="serum_creatinine",value=1.2, info="Choose serum_creatinine between 0 and 10")
serum_sodium = gr.Slider(100, 150, label="serum_sodium",value=142, info="Choose serum_sodium between 100 and 150")
sex =  gr.Radio(["0", "1"], label="sex",value="0", info="Sex ->  0-Female, 1-Male")
smoking = gr.Radio(["0", "1"], label="smoking",value="0", info="Does patient smoke ->  0-False, 1-True")
time = gr.Slider(0, 365, label="time",value=66, info="Choose time between 0 and 365")

# Output response
outputs = gr.Textbox(type="text", label='The patient survival predictions is :')


# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"


iface = gr.Interface(fn = predict_death_event,
                         inputs = [age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time],
                         outputs = [outputs],
                         title = title,
                         description = description,
                         allow_flagging='never')

# iface.launch(share = True,debug=True,server_name="0.0.0.0", server_port = 8001)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


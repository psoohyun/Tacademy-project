
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os, re
from utils import *
from schema import Data, ModelInput
from typing import List, Dict
from predict_module import Predict
from pydantic import BaseModel
import pickle, numpy as np
import pandas as pd
from google.cloud import storage



app = FastAPI()


@app.get('/')
def activate():
    """
    :params:
    :return(return type): 현재의 로컬 위치와 연동된 gcp의 bucket(dict) 
    """
    f = os.getcwd()
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    return {"path":f, 'bucket':buckets}


@app.get('/model/info/')
def info():
    """
    :params:
    :return(return type):현재 저장된 모든 로컬 모델의 meta정보(Json)
    """
    a = show_params()
    return JSONResponse(content=a)


@app.get('/model/run/')
def model_run():
    """
    학습을 시킨 후 현재 로컬에 저장된 모델 리스트 반환
    :params:
    :return(return type): 현재 로컬에 저장된 모델 리스트(List)
    """
    os.system('python ./MLflow/main.py --is_keras 0')
    b = show_model()
    return b



@app.get('/model/predict')
def data_predict(data: Data, model: str):
    """
    예측할 데이터를 집어넣고 결과를 보여주는 페이지
    :params data: 예측할 데이터
    :params model: 결과 예측시 사용할 모델의 uuid
    :return(return type): 예측 결과(List)
    """
    # print(**data.dict())
    # print(data)
    # print(data.dict())
    # print(pd.DataFrame(data))
    # models= "330ded0fb7ba462a881357ab456591f5"
    # results = {"Sex": [1,2,3,1], "Age_band":[1,2,3,4], "Pclass":[1,2,3,1]}
    # print(type(results))
    # print(results, type(results))
    # print(model)
    input = ModelInput(data).input()
    result = Predict(model).loaded_model(input).tolist()
    return result


@app.get("/model/{model}/predict")
def each_model_predict(model: str):

    result = model_load_from_local(model).predict([[1,1,1]]).tolist()

    return result


@app.get("/model/from_gcp")
def model_in_gcp():

    model_list = get_model_list_from_gcp()
    
    return model_list

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8003)




 
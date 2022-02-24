import mlflow
import pandas as pd
import os, re

class Predict:
    
    def __init__(self, model: str):
        """
        클래스 객체 생성시 모델의 주소 반환
        :params model: 실행시킬 모델의 uuid
        :return: 
        """
        self.logged_model = 'runs:/'+ model +'/ml_model'

    def loaded_model(self, data):
        """
        모델을 통하여 값 예측
        :params data: 예측할 데이터
        :return(return type): 예측값(리스트)
        """
        return mlflow.pyfunc.load_model(
            self.logged_model
        ).predict(pd.DataFrame(data))


if __name__ == "__main__":
    a = Predict("330ded0fb7ba462a881357ab456591f5")
    data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}
    print(a.loaded_model(data))
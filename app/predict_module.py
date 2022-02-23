import mlflow
import pandas as pd
import os, re

class Predict:
    
    def __init__(self, model: str):
        self.logged_model = 'runs:/'+ model +'/ml_model'

    def loaded_model(self, data):
        return mlflow.pyfunc.load_model(
            self.logged_model
        ).predict(pd.DataFrame(data))

# model_list=[]

# for i in os.listdir('./mlruns/0'):
#     if not re.search('yaml$',i):
#         # print(i)
#         model_list.append(i)

# data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}
# for i in model_list:
#     a = Predict(i)
#     print(a.loaded_model(data))

if __name__ == "__main__":
    a = Predict("330ded0fb7ba462a881357ab456591f5")
    data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}
    print(a.loaded_model(data))
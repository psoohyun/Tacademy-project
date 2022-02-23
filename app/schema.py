from pydantic import BaseModel
from typing import List, Dict

class Data(BaseModel):

    Sex: List[int]
    Age_band: List[int]
    Pclass: List[int]

class ModelInput():

    def __init__(self, data: Data):
        self.sex = data.Sex
        self.age_band = data.Age_band
        self.pclass = data.Pclass

    def input(self):
        return {"Sex": self.sex, "Age_band": self.age_band, "Pclass": self.pclass}


# if __name__ == "__main__":
#     # a = Data({"Sex":[1], "Age_band":[1], "Pclass":[1]})
#     a = Data(Sex=[1],Age_band=[1],Pclass=[1])
#     # b = ModelInput(a).input()
#     b= ModelInput(a)
#     b.input
#     print(b.input())
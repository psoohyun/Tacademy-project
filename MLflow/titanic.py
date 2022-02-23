import pandas as pd

from preprocess import TitanicPreprocess   # preprocess.py라는 파일에서 "TitanicPreprocess" 라는 class 가져옴

from config import PathConfig              # config.py라는 파일에서 "PathConfig" 라는 class 가져옴
                                           ## config 파일에는 Path, Env 에 대한 class가 들어있음 

from dataio import DataIOSteam             # dataio.py 라는 파일에서 "DataIOSteam" 이라는 class 가져옴

from model import TitanicModeling          # model.py 라는 파일에서 "TitanicModeling" 이라는 class 가져옴




class TitanicMain(TitanicPreprocess, PathConfig, TitanicModeling, DataIOSteam):  # Class가 모두 인자로 온다.
    def __init__(self):                                                          # 아래의 class 객체들에서 여러 def 함수들을 가져온다.
        
        TitanicPreprocess.__init__(self)                                         # pass라고 적혀있음
        PathConfig.__init__(self)                                                # 현재 작업 디렉토리 / 데이터 위치 경로 받아옴
        
        TitanicModeling.__init__(self)                                           # pass라고 적혀 있음
        DataIOSteam.__init__(self)                                               # "DataIOSteam" 라는 class에 "__init__" 자체가 없을 때: class를 선언하는 것과 같음 => 선언한 class의 함수들 사용


    def run(self, is_keras=0, n_estimator=100):                                  # run함수는 is_keras=0, n_estimator=100 이 default
        data = self._get_data(self.titanic_path)                                 # self가 모두 같기 때문에, 위에서 불러온 class의 함수들을 "self.~"을 통해 사용할 수 있다.
                                                                                 # config.py에서 PathConfig라는 class를 불러왔고, 그 안에 self.titanic_path가 있어서 불러올 수 있음
        data = self.run_preprocessing(data)                                      # preprocess.py 파일에서 TitanicPreprocess라는 class의 run_preprocessing 이라는 함수를 사용하여 전처리()
        X, y = self._get_X_y(data)                                               # dataio.py 파일에서 DataIOSteam이라는 class의  get_X_y 함수를 사용하여 데이터 불러오기
        
        if is_keras:                                                             # is_keras=0 이 default니까, keras를 기본 api로 사용하지 않음
            model, model_info = self.run_keras_modeling(X, y)
            return model, model_info
        
        else:
            model, model_info = self.run_sklearn_modeling(X, y, n_estimator)     # is_keras=0 이 default니까, else 아래 코드가 실행되므로 sklearn을 기본 api로 사용하겠다는 것
            return model, model_info
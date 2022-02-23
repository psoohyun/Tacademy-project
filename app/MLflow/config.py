import os


class PathConfig:
    def __init__(self):      
        self.project_path = os.getcwd()                            # 현재 작업 경로를 project_path에 초기화
        self.titanic_path = f"{self.project_path}/data/titanic"    # 받아온 작업경로(project_path)에서 data라는 폴더안에 titanic이라는 폴더의 경로를 titanic_path에 초기화
        

class EnvConfig:
    def get_gender_mapping_code(self):                             # gender labeling할 함수 정의
        gender_mapping_info = {
            'male' : 0,
            'female' : 1,
        }

        return gender_mapping_info                                 # dictionary 자체가 반환됨
    
    def get_column_list(self):
        columns_list = ['Sex', 'Age_band', 'Pclass']               # 내가 사용할 3가지 컬럼명 지정 => 나중에 predict 할 때도 3개의 값을 주고 predcition함
        return columns_list
import pandas as pd

class DataIOSteam:

    def _get_data(self, path):
        return pd.read_csv(f'{path}/train.csv')   # class에서 정의한 함수를 사용할 때, 인자로 path를 지정해주면 path에 있는 train.csv를 읽어들임
    
    def _get_X_y(self, data):                     # _get_X_y라는 함수를 사용할 때, data를 인자로 지정
        X = data[data.columns[1:]]                # 0번째(label) col을 제외하고 feature col만 가져옴
        X = X[['Sex', 'Age_band', 'Pclass']]      # feature중 원하는 col만 선택해서 최종적으로 X 변수에 초기화
        y = data['Survived']                      # y 변수에 label 초기화

        return X, y                               # X, y를 반환
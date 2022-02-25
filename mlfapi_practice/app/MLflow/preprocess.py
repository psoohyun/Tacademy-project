import numpy as np


class TitanicPreprocess:
    def __init__(self):
        pass

    def run_preprocessing(self, data):  # data = train.csv
        data = self._set_initial(data)  # 아래에서 정의한 함수 _set_initial 를 사용하여 전처리
        data = self._set_fill_na(
            data
        )  # 아래에서 정의한 함수 _set_fill_na 를 사용하여 결측치에 대해서 imputation
        data = self._set_feature(
            data
        )  # 아래에서 정의한 함수 _set_feature 를 사용하여 feature setting
        data = self._set_replace(
            data
        )  # 아래에서 정의한 함수 _set_replace 를 사용하여 labeling, 필요없는 컬럼 삭제

        return data

    def _set_fill_na(self, data):  # 결측치 imputation ==> 조건에 따라서 imputation
        data.loc[(data["Age"].isnull()) & (data["Initial"] == "Mr"), "Age"] = 33
        data.loc[(data["Age"].isnull()) & (data["Initial"] == "Master"), "Age"] = 5
        data.loc[(data["Age"].isnull()) & (data["Initial"] == "Mrs"), "Age"] = 36
        data.loc[(data["Age"].isnull()) & (data["Initial"] == "Miss"), "Age"] = 22
        data.loc[(data["Age"].isnull()) & (data["Initial"] == "Other"), "Age"] = 46
        data["Embarked"].fillna("S", inplace=True)

        return data

    def _set_initial(self, data):
        data["Initial"] = 0  # 0값을 가지는 Initial이라는 컬럼 생성
        data["Initial"] = data["Name"].str.extract(
            "([A-Za-z]+)\."
        )  # Name이라는 컬럼에 정규식표현 사용해서 이름 추출해서 Initial 컬럼에 할당
        data["Initial"].replace(
            [
                "Mlle",
                "Mme",
                "Ms",
                "Dr",
                "Major",
                "Lady",
                "Countess",
                "Jonkheer",
                "Col",
                "Rev",
                "Capt",
                "Sir",
                "Don",
                "Dona",
            ],  # 이렇게 쓰인 것을
            [
                "Miss",
                "Miss",
                "Miss",
                "Mr",
                "Mr",
                "Mrs",
                "Mrs",
                "Other",
                "Other",
                "Other",
                "Mr",
                "Mr",
                "Mr",
                "Other",
            ],  # 이렇게 변경
            inplace=True,
        )

        return data

    def _set_feature(self, data):
        data["Fare"] = data["Fare"].map(
            lambda i: np.log(i) if i > 0 else 0
        )  # Fare라는 컬럼의 값들이 0보다 크면 log를 취해주고 음수이면 0으로 취해라
        data["Age_band"] = 0  # Age_band라는 컬럼에 0값으로 초기화
        data["Alone"] = 0  # Alone이라는 컬럼에 0값으로 초기화
        data["Family_Size"] = 0  # Family_Size라는 컬럼에 0값으로 초기화

        data.loc[
            data["Age"] <= 16, "Age_band"
        ] = 0  # Age가 16세 이하이면               Age_band 컬럼에 0으로 초기화
        data.loc[
            (data["Age"] > 16) & (data["Age"] <= 32), "Age_band"
        ] = 1  # Age가 16세 초과 32세 이하이면     Age_band 컬럼에 1로 초기화
        data.loc[
            (data["Age"] > 32) & (data["Age"] <= 48), "Age_band"
        ] = 2  # Age가 32세 초과 48세 이하이면     Age_band 컬럼에 2로 초기화
        data.loc[
            (data["Age"] > 48) & (data["Age"] <= 64), "Age_band"
        ] = 3  # Age가 48세 초과 64세 이하이면     Age_band 컬럼에 3으로 초기화
        data.loc[
            data["Age"] > 64, "Age_band"
        ] = 4  # Age가 64세 초과이면               Age_band 컬럼에 4로 초기화

        data["Family_Size"] = (
            data["Parch"] + data["SibSp"]
        )  # Parch 컬럼의 값과 SibSp 컬럼의 값을 합하여 Family_Size라는 새로운 컬럼을 만들어서 할당
        data.loc[
            data.Family_Size == 0, "Alone"
        ] = 1  # 바로 위 코드에서 생성한 Family_Size 컬럼의 값이 0이면 Alone이라는 새로운 컬럼을 만들어서 1이라는 값으로 초기화

        return data

    def _set_replace(self, data):
        data["Sex"].replace(
            ["male", "female"], [0, 1], inplace=True
        )  # Sex라는 컬럼의        male, female   value를               0, 1로           labeling
        data["Embarked"].replace(
            ["S", "C", "Q"], [0, 1, 2], inplace=True
        )  # Embarked라는 컬럼의   S, C, Q        value를               0, 1, 2로        labeling
        data["Initial"].replace(
            ["Mr", "Mrs", "Miss", "Master", "Other"], [0, 1, 2, 3, 4], inplace=True
        )  # Initial이라는 컬럼의  Mr, Mrs, Miss, Master, Other value를 0, 1, 2, 3, 4로  labeling
        data.drop(
            ["Name", "Age", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True
        )  # 필요없는 컬럼 drop

        return data  # 모든 전처리가 끝나면 Model train에 사용할 data를 return

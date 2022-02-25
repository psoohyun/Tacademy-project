import mlflow

logged_model = "runs:/321c496c84324468841c1175e969ba70/ml_model"  # 내가 train시킨 모델의 URI( Uniform Resource Identifier(통합 자원 식별자)는 인터넷에 있는 자원을
#                           나타내는 유일한 주소이다. URI의 존재는 인터넷에서 요구되는 기본조건으로서
#                           인터넷 프로토콜에 항상 붙어 다닌다. )

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(
    logged_model
)  # mlflow Library에 있는 pyfunc의 load_model에 내가 train한 모델이 있는
# URI 인자를 지정하고, 이 모델을 loaded_model이라는 변수에 초기화

# Predict on a Pandas DataFrame.
import pandas as pd

# 예측할 데이터 작성
data = {"Sex": [1, 0, 1, 1], "Age_band": [1, 2, 1, 1], "Pclass": [1, 3, 3, 3]}

# mlflow.pyfunc.load_model(logged_model).predict(예측할 데이터)를 통해 예측결과를 print
print(loaded_model.predict(pd.DataFrame(data)))

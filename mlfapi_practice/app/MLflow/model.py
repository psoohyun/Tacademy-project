import sys

# Machine Learning
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# from lightgbm import LGBMClassifier                  -> LightGBM 생략


# Deep Learning
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


class TitanicModeling:
    def __init__(self):
        pass

    # sklearn API를 사용한 ML train
    def run_sklearn_modeling(self, X, y, n_estimator):
        model = self._get_rf_model(
            n_estimator
        )  # 아래에서 정의한 _get_rf_model 함수에서 RandomForestClassifier(n_estimators=n_estimator, max_depth=5)를 반환하므로 모델 객체를 model이라는 변수에 정의
        # lgbm_model = self._get_lgbm_model(n_estimator)

        model.fit(X, y)
        # lgbm_model.fit(X, y)

        model_info = (
            {  # model의 정보(score, parameter)를 담고있는 dictionary를 model_info라는 변수에 초기화
                "score": {"model_score": model.score(X, y)},
                "params": model.get_params(),  # model의 parameter를 보여줌
            }
        )

        return model, model_info

    # Tensorflow API를 사용한 DL train
    def run_keras_modeling(self, X, y):
        model = (
            self._get_keras_model()
        )  # 아래에서 정의한 _get_keras_model 함수에서 Dense를 쌓은 model을 반환하므로 이 모델 객체를 model이라는 변수에 정의
        model.fit(X, y, epochs=20, batch_size=10)  # fitting
        # predictions = model.predict(X)
        # print('keras prediction : ', predictions[:5])

        model_info = (
            {  # model의 정보(score, parameter)를 담고있는 dictionary를 model_info라는 변수에 초기화
                "score": {"model_score": np.float64(round(model.evaluate(X, y)[1], 2))},
                "params": {"epochs": 20, "batch_size": 10},
            }
        )

        return model, model_info

    # 모델 객체생성 하는 def들
    ## Random Forest
    def _get_rf_model(self, n_estimator):
        return RandomForestClassifier(n_estimators=n_estimator, max_depth=5)

    ## Light GBM
    # def _get_lgbm_model(self, n_estimator):
    #    return LGBMClassifier(n_estimators=n_estimator)

    ## DeepLearngin(Layer 쌓기)
    def _get_keras_model(self):
        inp = Input(shape=(3,), name="inp_layer")  # 0층 (input layer)
        dense_layer_1 = Dense(
            32, activation="relu", name="dense_1"
        )  # 1층 (hidden layer)
        dense_layer_2 = Dense(
            16, activation="relu", name="dense_2"
        )  # 2층 (hidden layer)
        predict_layer = Dense(
            1, activation="sigmoid", name="predict_layer"
        )  # 3층 (output layer)

        dense_vector_1 = dense_layer_1(inp)  # 1층 layer와 input layer(feature) matmul
        dense_vector_2 = dense_layer_2(
            dense_vector_1
        )  # 2층 layer와 1층 layer            matmul
        predict_vector = predict_layer(
            dense_vector_2
        )  # 3층 layer와 2층 layer            matmul

        model = Model(
            inputs=inp, outputs=predict_vector
        )  # "Model"이라는 tf(keras) 함수에 input(inp 변수)과 output(모든 matmul을 완료한 predict_vector 변수)을 지정
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["acc"]
        )  # 최적화
        return model  # 정의한 model을 반환

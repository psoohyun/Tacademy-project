# Library Imort
import argparse  # argparse는 cmd에서 ML Model로 인자의 값을 지정해 predict 할 때 사용해야 하므로 필요하다.
import sys


# 아래와 같이 mlflow는 수 많은 Python API로 ML/DL Library or Framework를 지원함(sklearn, tensorflow or pytorch etc.)
import mlflow
from mlflow import sklearn as ml_sklearn  # MLflow의 Python API
from mlflow import tensorflow  # MLflow의 ML Framework
from mlflow import log_artifacts  # MLflow에서 artifacts를 기록
from mlflow import (
    log_metric,
    log_metrics,
)  # MLflow에서 지원하는 metric을 저장하는 library -> -s가 붙으면 딕셔너리(key-value) 형태로 여러 metric을 저장
from mlflow import (
    log_param,
    log_params,
)  # MLflow에서 지원하는 parameter를 저장하는 library -> -s가 붙으면 딕셔너리(key-value) 형태로 여러 parameter를 저장


# py 파일로부터 생성된 class를 아래와 같이 import 할 수 있음
from titanic import TitanicMain  # titanic.py 파일에서 정의된 'TitanicMain()' class import


def _str2bool(v):
    if isinstance(v, bool):  # isinstance -> 입력되는 매개변수 v가 bool형인지 알아보는 것
        return v
    if v.lower() in (
        "yes",
        "true",
        "t",
        "y",
        "1",
    ):  # 입력되는 매개변수 v를 소문자로 바꿨을 때, (yes, true, ...)안에 들어 있다면 -> True 반환
        return True
    elif v.lower() in (
        "no",
        "false",
        "f",
        "n",
        "0",
    ):  # 입력되는 매개변수 v를 소문자로 바꿨을 때, (no, false ...)안에 들어 있다면 -> False 반환
        return False

    # 우리가 의도하지 않는 방향으로 돌아가는 것을 방지하기 위해서 일부러 에러를 발생시켜야 할 때가 있는데 이 때, raise를 씀
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected."
        )  # 위의 if, if, elif에서 걸러지면 "Boolean value expected."를 print 해라


# 모듈을 실행할 수 있는 방법은 1) interpreter에 직접 실행하거나, 2) import 하거나 ==> 이해를 돕기 위해 아래와 같은 파일이 있다고 하자
"""
//excuteThisModule.py
def func():
    print("function working")

if __name__ == "__main__":
    print("직접 실행")
    print(__name__)
else:
    print("임포트되어 사용됨")
    print(__name__)
"""

## 1) interpreter에 직접실행       : python excuteThisModule.py
##             ``          한 결과 : 직접 실행
##                                   __main__                        ==> interpreter에 직접 실행하면, __name__ 변수에 "__main__"이 담겨서 print 됨


## 2) 모듈에서 import해서 실행        : import excuteThisModule.py
##                                     excuteThisModule.func()
##             ``            한 결과 : 임포트되어 사용됨
##                                     executeThisModule             ==> 모듈에서 import해서 실행하면, __name__ 변수에 "excuteThisModule"이 담겨서 print 됨


# 우리는 interpreter에서 실행할 것이다. 위에서 설명한 것에 따라 if 아래에 있는 code들을 실행할 것이다. (아래 코드들은 나중에 Model을 train할 때 사용하려고 함)
if __name__ == "__main__":  # 파이썬에서 가장 일반적으로 사용하는 구문으로, 해당 어플리케이션이 모듈로 동작할 수 있도록 해준다.
    # mlflow.set_tracking_uri("192.168.1.41:5000")
    # mlflow.set_registry_uri('192.168.1.41:5000')
    # experiment_id = mlflow.get_experiment_by_name("aa")
    # if experiment_id is None:
    #     experiment_id = mlflow.create_experiment("aa")
    # else:
    #     experiment_id = experiment_id.experiment_id

    argument_parser = (
        argparse.ArgumentParser()
    )  # 인자값(argument)을 받을 수 있는 parser(인스턴스) 생성

    argument_parser.add_argument(  # add_argument() method를 통해 원하는 만큼 인자 종류를 추가
        "--is_keras",
        type=str,  # 나중에 train 할 때, "--is_keras 0"을 입력함으로써, keras를 사용하게 됨
        help="please input 1 or 0",
    )
    argument_parser.add_argument(  # add_argument() method를 통해 원하는 만큼 인자 종류를 추가
        "--n_estimator",
        type=int,
        default=100,  # 나중에 train 할 때, "--n_estimator 110"을 입력함으로써, 매개변수 n_estimator에 110을 지정할 수 있음 (default=100)
    )

    args = argument_parser.parse_args()  # 위의 내용들을 저장

    ## 예외처리
    try:
        is_keras = _str2bool(
            args.is_keras
        )  # try 아래 code를 실행중에 오류가 발생하면 ( is_keras에 0 or 1이 안들어가면 )
    except argparse.ArgumentTypeError as E:  # except 아래 code를 실행하라             ( print안의 문장을 실행시키고 )
        print("ERROR!! please input is_keras 0 or 1")
        sys.exit()  # exit()과 sys.exit()이 있는데
        # exit()은 cmd(shell)에서 쓰고, sys.exit()은 python(.py) 파일 안에서 사용 ==> 역할: 프로그램 종료

    titanic = (
        TitanicMain()
    )  # titanic.py에 정의된 'TitanicMain()' class를 'titanic' 이라는 변수에 초기화

    ## titanic.py에 정의된 'TitanicMain()' class에서 정의된 'run'이라는 함수의 매개변수 중 하나인 'is_keras'의 default 값 = 0 이다. => 기본적으로 sklearn을 사용하겠다는 뜻
    if is_keras:  # ==  is_keras=0 (default)
        with mlflow.start_run():

            tf_model, eval = titanic.run(
                is_keras
            )  # run이라는 함수 역할: data split, modeling(is_keras 매개변수에 따라서 --> keras ver. / sklearn)
            log_metric("tf keras score", eval)  # log_metric("Metric name", value)
            ml_sklearn.log_model(
                tf_model, "tf2_model"
            )  # log_model(model, 'Model name'): MLflow에서 Python API로 제공하는 sklearn의 model을 log_model로 기록
            print(
                "Model saved in run %s" % mlflow.active_run().info.run_uuid
            )  # 현재 활성 Run 상태 가져오기

        # log_artifacts('gs://psh_bucket/mlflow','ml_model')

    else:  # is_keras = 1

        with mlflow.start_run():

            model, model_info = titanic.run(
                is_keras, args.n_estimator
            )  # titanic class의 'run'이라는 함수 실행
            # args = argument_parser.parse_args()
            # argument_parser = argparse.ArgumentParser()

            # log_metric("rf_score", score_info['rf_model_score'])         # log_metric(metric_name(key), value)
            # log_metric("lgbm_score", score_info['lgbm_model_score'])

            # metrics 저장
            log_metrics(
                model_info["score"]
            )  # "model.py"라는 파일 안에, TitanicModeling 이라는 class에, run_sklearn_modeling 라는 함수 안에
            # model_info라는 딕셔너리가 있는데, 그 딕셔너리 안에 'score'라는 딕셔너리가 있는데, 그 안에 key:value를 가져오는 것이다.

            # parameters 저장
            ## for name, value in model_info['params'].items():
            log_params(
                model_info["params"]
            )  # "model.py"라는 파일 안에, TitanicModeling 이라는 class에, run_sklearn_modeling 라는 함수 안에
            # model_info라는 딕셔너리가 있는데, 그 딕셔너리 안에 'params'라는 key가 있는데, 그 key의 value에 딕셔너리가 있는데, 그 안에 key:value를 가져오는 것이다.

            # model 저장
            ml_sklearn.log_model(
                model, "ml_model"
            )  # log_model(model, 'Model name'): MLflow에서 Python API로 제공하는 sklearn의 model을 mlflow의 log_model을 사용해 기록
            print(
                "Model saved in run %s" % mlflow.active_run().info.run_uuid
            )  # 현재 활성 Run 상태 가져오기

        # log_artifacts('gs://psh_bucket/mlflow','ml_model')
        

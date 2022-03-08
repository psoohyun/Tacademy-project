import os

# Path를 구성하는 class


class PathConfig:
    def __init__(self):
        """
        # Description: 현재 작업경로를 얻고, 그 작업경로에서 data가 있는 경로를 지정해 줍니다.
        - project_path: 현재 작업 경로
        - titanic_path: train 시킬 data가 있는 경로
        -------------
        # Parameter
        -------------
        # Return
        : True or False
        """
        self.project_path = os.getcwd()
        # 받아온 작업경로(project_path)에서 data/pgdata/titanic 경로를 titanic_path에 초기화
        # json_server
        self.news_path = f"{self.project_path}/data/pgdata/news_json"
        # labeled_server
        self.labeled_path = f"{self.project_path}/data/pgdata/labeled_data"
        # model_server- 일단은 config에서 사용하지는 않지만, 사용할 수 잇을지 몰라서.
        # 이부분 수정 필요. 자유롭게 모델을 바꿀 수 있게.
        self.model_path = f"{self.project_path}/model/mobilebert.pt"
        # 차후에 server, gcp에서 받아오는 부분을 수정해야 한다.


# train 환경을 구성하는 class
class EnvConfig:
    def get_column_list(self):
        """
        # Description: train에 사용할 columns list를 반환합니다.
        - columns_list: train에 사용할 columns을 담은 list
        -------------
        # Parameter
        -------------
        # Return
        : columns_list인 list 자체를 반환
        """
        columns_list = [
            "date",
            "title",
            "content",
            "source",
            "url",
            "category",
        ]  # train에 사용할 columns 지정 => predict 할 때도 3개의 값을 주고 predcition 해야함
        return columns_list

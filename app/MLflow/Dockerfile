# syntax=docker/dockerfile:1

# Python 버전 선택
# 필요한 버전을 https://hub.docker.com/_/python 여기서 확인할 수 있습니다.
FROM python:3.7

# 디폴트 폴더 변경
WORKDIR /app/mlflow/

# Pip 패키지 설치
RUN pip3 install -r requirements.txt

# 필요한 파일을 복사
COPY ./ /app/mlflw/
CMD [ "python", "main.py", "-is_keras 0"]
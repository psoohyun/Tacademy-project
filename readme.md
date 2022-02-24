# Read Me

간단한 서비스를 위해 만든 Fast API  
 
## Description
머신 러닝 서비스 배포를 위한 Fast API

## Environment
```
fastapi==0.63.0  
google-cloud-storage==2.1.0  
mlflow==1.23.1  
pickle5==0.0.12  
pydantic==1.7.3  
uvicorn==0.13.3 
```
## Prerequisite
- **RabbitMQ**가 설치되어 있어야함



## Files
``` python
app ┬ MLflow
    ├ mlruns
    ├ main.py
    ├ schema.py
    ├ utils.py
    └ predict_module.py
```

## Usage
> python ./app/main.py
import os, re
from typing import List
import pickle5 as pickle
from google.cloud import storage

# mlruns 밑에 있는 결과물 폴더명 가져와서 model_list 안에 집어넣기
# ex) 9602e4000eef40d4847ce8f6d3c41eda <<이런 파일명들
def show_model(): 
    model_list = []

    for i in os.listdir('./mlruns/0'):
        if not re.search('yaml$',i):
            # print(i)
            model_list.append(i)
    
    return model_list



# 결과물 폴더 안에 각 모델의 model_score와 params 폴더 밑에 있는 parameter 값들을
# 가져와서 params 딕셔너리 안에 {모델명: {모델 파라미터1: 값1, ...}, ...} 형태로 집어넣기
def show_params():
    
    model_params = {}
    model_list = show_model()

    for i in range(len(model_list)):

        # 폴더 params 밑에 있는 값들 집어 넣기
        path_params = './mlruns/0/'+model_list[i]+'/params/'
        content = {}
        for j in os.listdir(path_params):
            with open(path_params+j) as f:
                content[j] = f.readlines()[0]

        # 폴더 metrics 밑에 있는 값들 집어 넣기
        path_metrics = './mlruns/0/'+ model_list[i] +'/metrics/'
        for k in os.listdir(path_metrics):
            with open(path_metrics+k) as g:
                content[k] = g.readlines()[0].split()[1]
        
        content['model_name'] = model_list[i]
        model_params[i+1] = content
    
    return model_params


# 로컬 저장소에서 모델 불러오기
def model_load_from_local(model: str):

    path = f"./mlruns/0/{model}/artifacts/ml_model/model.pkl"
    with open(path,'rb') as f:
        ai_model = pickle.load(f)

    return ai_model




# GCP에서 모델 목록 가져오기
def get_model_list_from_gcp():
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/TFG5076XG/practice/cloud_practice/chromatic-hue-341712-06f40be9bd11.json"

    bucket_name = 'hjh_bucket'
    client = storage.Client()
    blobs = client.list_blobs(bucket_name)

    model_list=[]

    for i in blobs:
        # print(i.name)
        try:
            if (len(i.name.split('/')[2])>=20)&(i.name.split('/')[2] not in model_list):
                model_list.append(i.name.split('/')[2])
        except: pass

    return model_list

# GCP에서 모델을 가져오기
def get_pkl_from_gcp(model: str):

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/TFG5076XG/practice/cloud_practice/chromatic-hue-341712-06f40be9bd11.json"

    bucket_name = 'hjh_bucket'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    source_path = f'titanic/1/{model}/artifacts/ml_model/model.pkl'
    blobs = bucket.blob(source_path)

    model_bytes = blobs.download_as_bytes(raw_download=True)

    # print(model_bytes)

    # with open(model_bytes,'rb') as f:
    #     ai_model = pickle.Unpickler.load(f)

    ai_model = pickle.loads(model_bytes, encoding="bytes")

    return ai_model
  



if __name__ == "__main__":
    a = get_model_list_from_gcp()
    # print(a)
    b = get_pkl_from_gcp('2e93bc3a56f14d10aaad7bcdebd9f61e')
    print(b.predict([[1,1,1]]).tolist())
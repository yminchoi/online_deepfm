# Databricks notebook source
import json
import os
import sys
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from pyspark.sql.functions import col

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from sklearn.preprocessing import LabelEncoder


sys.path.append(
    "/Workspace/Repos/dataprd-repos/dataprd-job-legacy/bin/python_script/common/utils"
)
sys.path.append("/Workspace/Users/youngmin.choi@musinsa.com/online_deepfm/models/DeepCTR-Torch")



from models import *

import boto3
import io
import pickle
from io import BytesIO


from deepctr_torch.inputs import (
    SparseFeat,
    DenseFeat,
    get_feature_names,
    VarLenSparseFeat,
    feature_format_deepfm
)
from deepctr_torch.models import *
from deepctr_torch.mss_utils import * 

sys.path.append("/Workspace/Repos/dataprd-repos/dataprd-spi-on-bricklane/src")

#from dags.util.utils import InputParser
# from get_banner_model_config import *

# COMMAND ----------

# MAGIC %md 
# MAGIC # Inferecne Data 호출

# COMMAND ----------

# date
dt = getArgument("date", "20241030")  # YYYYMMDD
dt_datetime = datetime.strptime(dt, "%Y%m%d")
dt_pre_7days = (dt_datetime - timedelta(days=7)).strftime("%Y%m%d")

# dt = '20241125'
year = dt[:4]
YM = dt[:6]
DT = dt[:8]

HOUR = getArgument("hour", "08")
# HOUR = "16"
MIN = "00"

start_date = dt_pre_7days
end_date = dt
current_time = dt + HOUR + MIN

print("year", year)
print("dt", DT)
print("hour", HOUR)
print("current_time", current_time)

# store
store = "musinsa"
print("store", store)

# S3 path
get_banner_model_configs = get_banner_model_configs()
print(get_banner_model_configs)

model_name = 'deepfm'
model_type = 'personalized'
s3_load_path = get_banner_model_configs[model_name]["personalized"]["save_path_prefix"]
s3_save_path = get_banner_model_configs[model_name][model_type]["save_path_prefix"]
print("s3_load_path: {}, s3_save_path: {}".format(s3_load_path, s3_save_path))

s3_bucket = "musinsa-data"
print("s3_bucket: " + s3_bucket)

s3_client = boto3.client('s3')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # model config 

# COMMAND ----------

model_params = get_model_params()
embedding_dim = model_params.get('embed_dim', 8)
mlp_dims = model_params.get('mlp_dims', [16, 8])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference Data load

# COMMAND ----------


paarquet_load_path = "/dbfs/FileStore/ymchoi/banner_campaign_test/inference_data.parquet"
data = pd.read_parquet(paarquet_load_path)
display(data)


data = data.assign(
    key=data["uid"],
    value=data["banner_campaign_ad_group_display_type_contents_id"]
)

sparse_features, dense_features = feature_selection()
print(sparse_features)
print(dense_features)


# COMMAND ----------

# MAGIC %md
# MAGIC # encoder load 

# COMMAND ----------

import pickle
# 불러오기
def load_encoders(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# 2. 불러오기
encoders = load_encoders('encoders.pkl')
print(encoders)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature format 변경

# COMMAND ----------


# def feature_format_deepfm(data, encoders, sparse_features, dense_features, embedding_dim):

#     print(f"5. feature embedding - embedding size {embedding_dim}")
    
#     spar_feat_list = [
#         SparseFeat(
#             feat,
#             vocabulary_size=len(encoders[feat].classes_) + 1,
#             embedding_dim=embedding_dim
#         )
#         for feat in sparse_features
#     ]
    
#     dense_feat_list = [DenseFeat(feat, 1, ) for feat in dense_features]
#     fixlen_feature_columns = spar_feat_list + dense_feat_list

#     dnn_feature_columns = fixlen_feature_columns
#     linear_feature_columns = fixlen_feature_columns
#     feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

#     return dnn_feature_columns, linear_feature_columns, feature_names

# COMMAND ----------

data = feature_encoding(data, sparse_features, dense_features, encoders)

dnn_feature_columns, linear_feature_columns, feature_names = feature_format_deepfm(data, encoders, sparse_features, dense_features, embedding_dim)


data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # model load

# COMMAND ----------

model = torch.load('deepfm_model.pth')
model.eval()

model 

# COMMAND ----------

# MAGIC %md
# MAGIC # data tensor로 변경 후 infernece 

# COMMAND ----------

import torch.utils.data as Data


input_features = {name: data[name] for name in feature_names}
x=input_features

if isinstance(x, dict):
    x = [x[feature] for feature in model.feature_index]
for i in range(len(x)):
    if len(x[i].shape) == 1:
        x[i] = np.expand_dims(x[i], axis=1)

tensor_data = Data.TensorDataset(
    torch.from_numpy(np.concatenate(x, axis=-1)))
test_loader = DataLoader(
    dataset=tensor_data, shuffle=False, batch_size=batch_size)

pred_ans = []
with torch.no_grad():
    for _, x_test in enumerate(test_loader):
        print(x_test)
        x = x_test[0].to(model.device).float()

        y_pred = model(x).cpu().data.numpy()  # .squeeze()
        pred_ans.append(y_pred)


# COMMAND ----------



# COMMAND ----------



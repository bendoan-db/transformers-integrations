import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
import os

def get_hf_token(secret_scope:str=None, secret_key:str=None, use_env_variable=False):
    if use_env_variable:
        hf_token = os.environ['HF_TOKEN']
    else:    
        try:
            hf_token = dbutils.secrets.get(scope = secret_scope, key = secret_key)
            print("Using Token with scope: " + secret_scope +" and key: " + secret_key)
        except Exception as error:
            hf_token = None
            print("Exception: " + str(error))
            print("No token found with this scope + key.")
    return hf_token

def log_gpu_metrics_mlflow(run_type:str, step):
   for i in range(torch.cuda.device_count()):
     mlflow.log_metric(run_type+"_gpu_utilization_gb_rank_"+str(i)+"_pct", Decimal(torch.cuda.utilization(device=i)/100), step=step)


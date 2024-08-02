import os
import yaml
from functools import lru_cache
from .common.colors import get_color_dict
import warnings
import pandas as pd
import torch

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", 'data')

model_configs_path = os.path.join(DATA_PATH, "supported_models.yml")

@lru_cache(maxsize=50)
def get_supported_models(config_path: str = model_configs_path) -> dict:
    ''' 返回支持的模型配置信息 '''
    with open(config_path, "r") as f:
        models_configs = yaml.safe_load(f)
    return models_configs

def get_model_names() -> list:
    ''' 返回支持的模型名称列表 '''
    models_configs = get_supported_models()
    return list(models_configs.keys())

def get_model_keys() -> list:
    ''' 返回模型的配置信息 '''

    models_configs = get_supported_models()
    keys = []
    for _, v in models_configs.items():
        for key, _ in v.items():
            if key not in keys:
                keys.append(key)
    return keys

@lru_cache(maxsize=50)
def get_model_info(model_name:str, model_key:str)->str:
    ''' 返回指定模型的指定标签的指定信息 '''
    models_configs = get_supported_models()
    model_config = models_configs.get(model_name)
    assert model_config is not None, f"Model {model_name} is not supported."
    model_info_dict = model_config.get(model_key, None)
    return model_info_dict

@lru_cache(maxsize=50)
def get_predictor(model_name:str, key:str):
    if model_name is None or key is None:
        return None
    from macls.predict import MAClsPredictor
    
    info = get_model_info(model_name, key)
    if info is None:
        warnings.warn(f"Model {model_name} does not have {key} information.")
        return None
    config_path = os.path.join(DATA_PATH, info.get('configs_path'))
    model_path = os.path.join(DATA_PATH, info.get('model_path'))
    label_list_path = os.path.join(DATA_PATH, info.get('label_list_path'),'label_list.txt')
    
    ''' 返回指定模型的预测器 '''
    return MAClsPredictor(configs=config_path, model_path=model_path, label_list_path=label_list_path, use_gpu=torch.cuda.is_available())

@lru_cache(maxsize=50)
def get_labels_info(model_name:str, key:str)->str:
    ''' 返回指定模型的标签信息 '''
    info = get_model_info(model_name, key)
    if info is None:
        warnings.warn(f"Model {model_name} does not have {key} information.")
        return None
    labels_info_path = os.path.join(DATA_PATH, info.get('label_list_path'),'label_dataframe.csv')
    if not os.path.exists(labels_info_path):
        return None
    df = pd.read_csv(labels_info_path)
    df['parent'] = df['sources'].apply(lambda x: x.split('parent')[1].split('"')[1])
    labels_info = dict(zip(df['name'], df['parent']))
    return labels_info

def get_labels(model_name:str, key:str):
    info = get_model_info(model_name, key)
    label_list_path = os.path.join(DATA_PATH, info.get('label_list_path'),'label_list.txt')
    with open(label_list_path, 'r') as f:
        labels = f.readlines()
    return [l.strip() for l in labels if l.strip()]

def get_color_map(lst:list, random_state=7)->dict:
    ''' 返回指定模型的颜色映射 '''
    color_dict = get_color_dict(lst=lst, random_state=random_state)
    return {k:v['hex'] for k, v in color_dict.items()}
    



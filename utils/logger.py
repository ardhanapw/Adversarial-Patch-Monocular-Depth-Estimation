import json
import comet_ml
from torch.utils.tensorboard import SummaryWriter

def init_tensorboard(log_path):
    return SummaryWriter(log_path)

def start_comet_ml(file_path, model_name):
    with open(file_path, "r") as f:
        comet_ml_cred = json.load(f)
        
    return comet_ml.start(api_key=comet_ml_cred["api_key"], project_name=f"Generate adv patch on {model_name}")

#experiment = start_comet_ml('test.json')
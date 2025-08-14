import torch
import argparse
import os

from utils.cfg_loader import load_yaml
import utils.logger

from models import load_models
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default='patch_trainer.yml')

def main():
    #arg parser and file config
    args = parser.parse_args()
    cfg = load_yaml(args.config_path)
    
    #create destination directory
    os.makedirs(cfg["log"]["tensorboard_dir"], exist_ok=True)
    os.makedirs(cfg["log"]["patch_checkpoint_dir"], exist_ok=True)
    
    #Tensorboard
    writer = utils.logger.init_tensorboard(cfg['log']['tensorboard_dir'])
    for key, value in cfg.items():
        writer.add_text(key, str(value))
    
    #initialize device
    device = torch.device(cfg['device'])

    #load model
    model_name, model_path = cfg['model']['model_name'], cfg['model']['model_path']
    initialize_model = load_models(model_name, model_path, device)
    
    #inference test
    img = Image.open('sample for inference/test_scene1.jpg')
    prediction = initialize_model.predict(img) #W, H in PIL, but torch expects H, W
    print(prediction.shape)
    initialize_model.plot(img, prediction, save=True, save_path='sample for inference/test_scene1_res.jpg')
    
if __name__ == "__main__":
    main()
    
    
    
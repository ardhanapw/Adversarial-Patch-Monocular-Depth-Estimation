import torch
import argparse

from utils.cfg_loader import load_yaml

from models import load_models
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default='config.yml')

def main():
    #arg parser and file config
    args = parser.parse_args()
    cfg = load_yaml(args.config_path)
    
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
    
    
    
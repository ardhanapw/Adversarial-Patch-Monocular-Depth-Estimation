import comet_ml
import torch
import torch.optim as optim
from torchvision import transforms

import argparse
import os

from utils.cfg_loader import load_yaml
from utils.data_loader import LoadFromImageFile
import utils.custom_transforms as transformer
import utils.logger

from models import load_models
from adv_utils.loss import AdversarialLoss
from adv_utils.tasks import AdvPatchTask
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default='patch_trainer.yml')

def main():
    #arg parser and file config
    args = parser.parse_args()
    cfg = load_yaml(args.config_path)
    #print(cfg)
    
    #create destination directory
    os.makedirs(cfg["log"]["tensorboard_dir"], exist_ok=True)
    os.makedirs(cfg["log"]["patch_checkpoint_dir"], exist_ok=True)
    
    #Comet ML
    experiment = utils.logger.start_comet_ml(cfg['log']['comet_ml_credentials'], cfg['model']['model_name'])
    experiment.log_text(cfg, step=0)
    
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
    #img = Image.open('sample for inference/test_scene1.jpg')
    #prediction = initialize_model.predict(img, (img.size[1], img.size[0])) #W, H in PIL, but torch expects H, W
    #print(prediction) 
    #initialize_model.plot(img, prediction, save=True, save_path='sample for inference/test_scene1_res.jpg')
    
    
    #dataloader and augmentation
    train_transform = transformer.Compose([
        transformer.RandomHorizontalFlip(),
        transformer.RandomAugumentColor(),
        transformer.RandomScaleCrop(scale_range=(0.85, 1.0)),
        #transformer.ResizeImage(h=args.height, w=args.width),
        transformer.ArrayToTensor()
    ])
    
    train_set = LoadFromImageFile(
        data_path=cfg['dataset']['dataset_path'],
        filenames_file=cfg['lists']['train_list'],
        seed=0,
        transform=train_transform,
        extension='.jpg'
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=cfg['dataset']['batch_size'],
        shuffle=True,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    brightness = cfg['patch']['patch_augmentation']['brightness']
    contrast = cfg['patch']['patch_augmentation']['contrast']
    output_augmentation = transforms.Compose(
        [
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
        ]
    )
    
    #adversarial loss
    adversarial_loss = AdversarialLoss(
        disp_loss_weight = cfg['patch']['loss_weight']['disp_loss'],
        nps_loss_weight = cfg['patch']['loss_weight']['nps_loss'],
        tv_loss_weight = cfg['patch']['loss_weight']['tv_loss'],
        nps_triplet_scores_fpath= cfg['lists']['30_rgb_triplets_path']
    )
    
    #initialize patch
    adv_patch_cpu = torch.rand((3, cfg['patch']['resolution'], cfg['patch']['resolution'])).to(device)
    adv_patch_cpu.requires_grad_(True)
    
    #optimizer
    optimizer = optim.Adam([adv_patch_cpu], lr=cfg['hyperparameter']['learning_rate'])
    
    #train patch
    adv_patch_trainer = AdvPatchTask(
        optimizer=optimizer,
        mde_model=initialize_model,
        loss=adversarial_loss,
        adv_patch=adv_patch_cpu,
        output_augmentation=output_augmentation,
        device=device,
        target_disp=cfg['patch']['target_disp_sigmoid']
    )
    
    #adv_patch_trainer.visualize(train_loader)
    #adv_patch_trainer.inference_test(train_loader)
    adv_patch_trainer.train(
        epochs=cfg['hyperparameter']['num_epoch'],
        train_dataset=train_loader,
        writer=writer,
        log_interval=cfg['log']['log_interval'],
        patch_export_path=cfg["log"]["patch_checkpoint_dir"]
    )
    
if __name__ == "__main__":
    main()
    
    
    
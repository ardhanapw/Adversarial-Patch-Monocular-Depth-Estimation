import torch
import argparse

from adv_utils.tasks import AdvPatchTask

from utils.cfg_loader import load_yaml
from utils.original_kitti_dataloader import KITTIRAWDataset, readlines
from utils.data_loader import load_patch_from_img

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
    
    #dataloader
    eval_files = readlines(cfg['lists']['eval_list'])
    eval_dataset = KITTIRAWDataset(cfg['dataset']['eval_path'], eval_files,
                                   192, 640, [0], 4, is_train=False)
    
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=cfg['dataset']['batch_size'],
        shuffle=True,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    #load patch
    if cfg['patch']['path'] == 'random':
        adv_patch_cpu = torch.rand((3, cfg['patch']['resolution'], cfg['patch']['resolution'])).to(device)
    else:
        adv_patch_cpu = load_patch_from_img(cfg['patch']['path']).to(device)
    
    #train patch
    #add target size to be tested
    adv_patch_instance = AdvPatchTask(
        mde_model=initialize_model,
        adv_patch=adv_patch_cpu,
        device=device,
    )
    
    #evaluate patch
    adv_patch_instance.visualize_adv(
        eval_dataset=eval_loader
    )
    
    #adv_patch_instance.evaluate(
    #    eval_dataset=eval_loader
    #)
    """
    adv_patch_trainer.train(
        epochs=cfg['hyperparameter']['num_epoch'],
        train_dataset=train_loader,
        writer=writer,
        log_interval=cfg['log']['log_interval'],
        patch_export_path=cfg["log"]["patch_checkpoint_dir"]
    )
    """
    
if __name__ == "__main__":
    main()
    
    
    
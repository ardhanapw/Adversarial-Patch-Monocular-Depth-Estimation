import os
import torch
import json
import random
import pandas as pd
import numpy as np
import cv2

import time

from PIL import Image
from tqdm import tqdm

from typing import Optional

from adv_utils.loss import AdversarialLoss
from models.model_mde import ModelMDE

from adv_utils.disp_converter import disp_to_depth
from adv_utils.patch import apply_patch
from utils.format_time import format_time


class AdvPatchTask:
    def __init__(
        self,
        mde_model: ModelMDE,
        optimizer=None,
        loss: AdversarialLoss=None,
        adv_patch=None,
        output_augmentation=None,
        device=None,
        target_disp: float=0.999999
    ):
        self.optimizer = optimizer
        self.mde_model = mde_model
        self.loss = loss
        self.adv_patch = adv_patch
        self.output_augmentation = output_augmentation
        self.device = device
        self.target_disp = target_disp
        
    def forward(
        self,
        task: str,
        batch, 
        adv_patch,
        adversarial_losses: Optional[AdversarialLoss] = None, 
    ):
        rgb = batch['left'].to(self.device)

        current_batch_size = rgb.shape[0]

        #apply and augment patch
        final_images, patch_masks = apply_patch(
            adv_patch, rgb, current_batch_size
        )

        #augment scene
        augmented_final_images = torch.stack(
            [
                self.output_augmentation(final_images[i])
                for i in range(current_batch_size)
            ]
        )

        if task == 'predict':
            predicted_disp = self.mde_model(augmented_final_images)
            #print(torch.min(predicted_disp), torch.max(predicted_disp))

            losses = adversarial_losses(
                adv_patch,
                patch_masks,
                predicted_disp,
                torch.full_like(predicted_disp, self.target_disp), #zero target disparity means far away object, vice versa
            ) #avoid zero or one target_disp on BCE loss

            return {
                "loss": losses,
                "final_images": final_images,
                "augmented_final_images": augmented_final_images,
                "patch_masks": patch_masks,
                "predicted_disp": predicted_disp,
            }
            
        elif task == 'visualize':
            return augmented_final_images
        
    def train(
        self,
        epochs:int,
        train_dataset,
        writer,
        log_interval:int,
        patch_export_path
    ):
        start_time = time.time()
        for epoch in range(epochs):
            ep_loss, ep_disp_loss, ep_nps_loss, ep_tv_loss = 0, 0, 0, 0
            
            for i_batch, sample in tqdm(enumerate(train_dataset), desc=f"Train Epoch {epoch+1}/{epochs}", total=len(train_dataset)):
                with torch.autograd.detect_anomaly():
                    self.optimizer.zero_grad()
                    results = self.forward('predict', sample, self.adv_patch, self.loss)
                    
                    #backpropagate
                    results['loss']["total_loss"].backward()
                    self.optimizer.step()
                    self.adv_patch.data.clamp_(0, 1)
                    
                    #print(results['loss'])
                    ep_loss += results['loss']['total_loss'].detach().cpu().numpy()
                    ep_disp_loss += results['loss']['disp_loss'].detach().cpu().numpy()
                    ep_nps_loss += results['loss']['nps_loss'].detach().cpu().numpy()
                    ep_tv_loss += results['loss']['tv_loss'].detach().cpu().numpy()

                    if i_batch % int(log_interval/train_dataset.batch_size) == 0:
                        iteration = len(train_dataset) * epoch + i_batch
                        writer.add_scalar("total_loss", results['loss']['total_loss'], iteration)
                        writer.add_scalar("loss/disp_loss", results['loss']['disp_loss'], iteration)
                        writer.add_scalar("loss/nps_loss", results['loss']['nps_loss'], iteration)
                        writer.add_scalar("loss/tv_loss", results['loss']['tv_loss'], iteration)
                        writer.add_scalar("misc/epoch", epoch, iteration)
                        writer.add_scalar("misc/learning_rate", self.optimizer.param_groups[0]["lr"], iteration)
                        writer.add_image("patch", self.adv_patch.detach().cpu().numpy(), iteration)
                    torch.cuda.empty_cache()
                    
            ep_loss = ep_loss/len(train_dataset)
            ep_disp_loss = ep_disp_loss/len(train_dataset)
            ep_nps_loss = ep_nps_loss/len(train_dataset)
            ep_tv_loss = ep_tv_loss/len(train_dataset)
            total_time = time.time() - start_time
            print('===============================')
            print(f"Total training time: {format_time(int(total_time))}")
            print('Epoch: ', epoch)
            print('Total epoch loss: ', ep_loss)
            print('Disparity loss: ', ep_disp_loss)
            print('NPS loss: ', ep_nps_loss)
            print('TV loss: ', ep_tv_loss)
            print('===============================')
            np.save(patch_export_path + '/epoch_{}_patch.npy'.format(str(epoch)), self.adv_patch.data.detach().cpu().numpy())
            np.save(patch_export_path + '/epoch_{}_mask.npy'.format(str(epoch)), results['patch_masks'].data.detach().cpu().numpy())
    def train_old(
        self,
        epochs:int,
        dataset,
        train_total_batch,
        val_total_batch,
        log_prediction_every:int,
        log_name:str
    ):
        # Check if direectory is exist or not, if ot make dirs
        dir_path = os.path.join('log', log_name)
        if os.path.exists(dir_path) is False:
            os.makedirs(dir_path)
        
        mean_value_train_list, mean_value_eval_list = [], []
        
        for epoch in range(epochs):
            mean_value_train, mean_value_eval = 0, 0
            step_train, step_val = 0, 0
            # Train iteration
            with tqdm(
                dataset['train'], total=train_total_batch, desc=f"Train Epoch {epoch + 1}/{epochs}"
            ) as train_pbar:
                for batch in train_pbar:
                    self.optimizer.zero_grad()

                    results = self.forward('predict', batch, self.adv_patch, self.loss)

                    # Backpropagate the loss
                    results['loss']["total_loss"].backward()
                    self.optimizer.step()

                    self.adv_patch.data.clamp_(0, 1)

                    # Update the progress bar
                    train_pbar.set_postfix(
                        {
                            key: value.item() if isinstance(value, torch.Tensor) else value
                            for key, value in results['loss'].items()
                            if 'loss' in key
                        }
                    )

                # Append train log
                mean_value_train_list.append(mean_value_train/step_train)
                
                # Save the texture as image
                Image.fromarray((self.adv_patch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(dir_path, f"texture_epoch_{epoch}.png")
                )
    
    def inference_test(
        self,
        dataset
    ):
        for i_batch, samples in enumerate(dataset):
            results = self.forward('predict', samples, self.adv_patch, self.loss)
            print(results)
            torch.cuda.empty_cache()
    
    def visualize(
        self,
        dataset,
        ):
        
        for i_batch, samples in enumerate(dataset):
            samples_augmented = self.forward('visualize', samples, self.adv_patch)
            
            for sample in samples_augmented:
                sample = sample.permute(1, 2, 0).detach().cpu().numpy()
                #print(np.max(sample))
                #sample = (sample * 255).astype(np.uint8)
                
                cv2.imshow('image', sample)
                cv2.waitKey(0)
                
            torch.cuda.empty_cache()
            
        cv2.destroyAllWindows()
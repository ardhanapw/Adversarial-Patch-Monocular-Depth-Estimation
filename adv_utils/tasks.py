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
from utils.original_kitti_dataloader import batch_post_process_disparity, compute_errors
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
        target_disp: float=0.999999,
        target_size=None
    ):
        self.optimizer = optimizer
        self.mde_model = mde_model
        self.loss = loss
        self.adv_patch = adv_patch
        self.output_augmentation = output_augmentation
        self.device = device
        self.target_disp = target_disp
        self.target_size = target_size
    
    def forward_eval(
        self,
        batch,
        adv_patch=None,
        target_size=None,
        mode='eval' #can be eval or inspect
    ):
        images = batch[("color", 0, 0)].to(self.device)
        xyxy = batch["label"].to(self.device)
        #car bbox -> for patch applying
        
        current_batch_size = images.shape[0]
        
        if adv_patch is not None:
            #apply patch
            final_images, patch_masks = apply_patch(
                adv_patch, images, self.target_size, current_batch_size, xyxy
            )
        
            _, adv_output = self.mde_model.predict(final_images, return_raw=True)
            
            #print("forward eval")
            #print(patch_masks.shape)
            #print(torch.unique(patch_masks))
            #print(torch.unique(patch_masks).numel())
            if mode == 'inspect':
                return final_images, adv_output
            
            elif mode == 'eval':
                _, benign_output = self.mde_model.predict(images, return_raw=True)
                return patch_masks, benign_output, adv_output
        
        _, output = self.mde_model.predict(images, return_raw=True)
        
        return output
        
    def forward(
        self,
        task: str,
        batch, 
        adv_patch,
        adversarial_losses: Optional[AdversarialLoss] = None, 
    ):
        #rgb = batch['left'].to(self.device)
        rgb = batch[("color", 0, 0)].to(self.device)
        xyxy = batch["label"].to(self.device)
        #use KITTI dataset for train loader first
        
        #print(rgb.shape)
        #print("forward")

        current_batch_size = rgb.shape[0]

        #apply and augment patch        
        final_images, patch_masks = apply_patch(
            adv_patch, rgb, self.target_size, current_batch_size, xyxy
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
    
    def evaluate(
        self,
        eval_dataset
    ):
        #min and max depth of KITTI dataset
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 8e+1
        
        predicted_benign_disp = []
        predicted_adv_disp = []
        gt_depths = []
        patch_masks = []
        
        with torch.no_grad():
            for batch in eval_dataset:
                patch_mask, benign_disp, adv_disp = self.forward_eval(batch, self.adv_patch, mode = 'eval')
                patch_mask = patch_mask.detach().cpu()[:, 0].numpy()
                #print(torch.mean(predicted_disp[("disp", 0)]))
                
                #print(predicted_disp[("disp", 0)].shape)
                #print(batch['depth_gt'].shape)
                
                #get predicted disparity on patch area
                #get ground truth disparity on path area
                predicted_benign_disp_scaled, _ = disp_to_depth(benign_disp[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
                predicted_benign_disp_scaled = predicted_benign_disp_scaled.detach().cpu()[:, 0].numpy()
                
                predicted_adv_disp_scaled, _ = disp_to_depth(adv_disp[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
                predicted_adv_disp_scaled = predicted_adv_disp_scaled.detach().cpu()[:, 0].numpy()
                
                #print("eval loop")
                #print(len(eval_dataset))
                #print(batch['depth_gt'].shape)
                #print(patch_mask.shape)
                #print(predicted_adv_disp_scaled.shape)
                
                #post processing (refer to monodepth)
                #print(predicted_depth.shape)
                #N = predicted_depth.shape[0] // 2
                #predicted_depth = batch_post_process_disparity(predicted_depth[:N], predicted_depth[N:, :, ::-1])
                gt_depths.append(batch['depth_gt'].detach().cpu()[:, 0].numpy())
                predicted_benign_disp.append(predicted_benign_disp_scaled)
                predicted_adv_disp.append(predicted_adv_disp_scaled)
                patch_masks.append(patch_mask)
                
        
        predicted_benign_disp = np.concatenate(predicted_benign_disp)
        predicted_adv_disp = np.concatenate(predicted_adv_disp)
        gt_depths = np.concatenate(gt_depths)
        patch_masks = np.concatenate(patch_masks)
        #print(predicted_depths)
        #print(predicted_depths.shape)
        #print(gt_depths.shape)
        
        errors_benign = []
        errors_adv = []
        ratios = []
        
        #benign phase
        for i in range(predicted_benign_disp.shape[0]):

            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_depth = predicted_benign_disp[i]
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
            pred_depth = 1 / pred_depth

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            #evaluasi yang hanya ditempel patch
            patch_masks_resized =  cv2.resize(
                patch_masks[i].astype(np.uint8), (gt_width, gt_height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mask = np.logical_and(mask, patch_masks_resized)
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= float(1)
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors_benign.append(compute_errors(gt_depth, pred_depth))
            
            ratios_1 = np.array(ratios)
            med = np.median(ratios_1)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios_1 / med)))

        mean_errors_benign = np.array(errors_benign).mean(0)
        
        print("Fase Benign")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_benign.tolist()) + "\\\\")
        print("\n-> Done!")
        
        #adv phase
        for i in range(predicted_adv_disp.shape[0]):

            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_depth = predicted_adv_disp[i]
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
            pred_depth = 1 / pred_depth

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            #evaluasi yang hanya ditempel patch
            patch_masks_resized =  cv2.resize(
                patch_masks[i].astype(np.uint8), (gt_width, gt_height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mask = np.logical_and(mask, patch_masks_resized)
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= float(1)
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors_adv.append(compute_errors(gt_depth, pred_depth))
            
            ratios_1 = np.array(ratios)
            med = np.median(ratios_1)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios_1 / med)))

        mean_errors_adv = np.array(errors_adv).mean(0)

        print("Fase Adversarial")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_adv.tolist()) + "\\\\")
        print("\n-> Done!")
        
    
    def inference_test(
        self,
        dataset
    ):
        for i_batch, samples in enumerate(dataset):
            results = self.forward('predict', samples, self.adv_patch, self.loss)
            print(results)
            torch.cuda.empty_cache()
    
    def visualize_benign(
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
        
    def visualize_adv(
        self,
        eval_dataset,
        ):
        
        for i_batch, samples in enumerate(eval_dataset):
            imgs = samples[("color", 0, 0)]
            
            if self.adv_patch is None:
                predictions = self.forward_eval(samples, self.adv_patch, mode='inspect')[("disp", 0)]
            else:
                imgs, outputs = self.forward_eval(samples, self.adv_patch, mode='inspect')
                predictions = outputs[("disp", 0)]
            
            for i_sample, (img, prediction) in enumerate(zip(imgs, predictions)):
                img = img.permute(1, 2, 0).detach().cpu().numpy()
                number = (i_batch+1) * (i_sample+1)
                self.mde_model.plot(img, prediction, save=True, save_path=f'sample for inference/test_{number}.jpg')
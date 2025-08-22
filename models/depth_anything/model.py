import os
import cv2
from models.depth_anything.networks.dpt import DepthAnything
from models.depth_anything.networks.util.transform import Resize, NormalizeImage, PrepareForNet
import torch
import torchvision.transforms as T

import torch.cuda.amp as amp

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

from models.model_mde import ModelMDE

import time

class CustomDepthAnything(ModelMDE):
    def __init__(self, model_path, device=None, **kwargs):
        self.device = device
        """
        self.transform = T.Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        """
        self.transform = T.Compose([
            T.Resize((518, 518)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        
        super(CustomDepthAnything, self).__init__(device=self.device, model_path=model_path, **kwargs)
    
    def load(
        self, model_path=None, device=None, **kwargs
    ):  
        encoder = model_path.split('/')[2]
        feed_height, feed_width = 518, 1722
        weight_path = os.path.join(model_path, f'depth_anything_{encoder}14.pth')
        self.depth_anything = DepthAnything(self.model_configs[encoder])
        self.depth_anything.load_state_dict(torch.load(weight_path))
        self.depth_anything.to(device)
        self.depth_anything.eval()
        
        return self.depth_anything, feed_height, feed_width
    
    def predict(self, img, return_raw=False):
        if not torch.is_tensor(img):
            img = T.ToTensor()(img).unsqueeze(0)
        
        H, W = int(img.shape[-2]), int(img.shape[-1])
        """
        imgs = []
        #preprocess from depth anything only works if img is converted to cv2 first
        for sample in img:
            sample_np = sample.detach().permute(1, 2, 0).cpu().numpy()
            transformed = self.transform({'image': sample_np})['image']
            imgs.append(torch.from_numpy(transformed))
        
        img_tensor = torch.stack(imgs).to(self.device)
        """
        img = self.transform(img)
        #start_time = time.time()
        #with torch.no_grad():
        with amp.autocast(enabled=True):
            output = self.depth_anything(img)
        #print(time.time() - start_time)

        output = torch.nn.functional.interpolate(
            output[None], (H, W), mode="bilinear", align_corners=False
        )
        output = (output - output.min()) / (output.max() - output.min())
        
        #[1, B, H, W] -> [B, 1, H, W]
        output = output.permute(1, 0, 2, 3)
        
        #depth anything has no feature extraction weight
        if return_raw:
            dummy = None
            return dummy, {("disp", 0): output}
        
        return output
        
    def visualize(self, prediction):
        disp_resized_np = prediction.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)

        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        
        return colormapped_im
    
    def plot(self, image, prediction, save=False, save_path=None):
        disp_resized_np = prediction.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)

        fig = plt.figure(figsize=(5, 5))
        plt.subplot(211)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(212)
        plt.imshow(disp_resized_np, cmap="magma", vmax=vmax)
        plt.title("Disparity")
        plt.axis("off")

        if save:
            plt.savefig(save_path)
        
        return fig
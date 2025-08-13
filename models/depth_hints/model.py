import os
from models.depth_hints.networks import ResnetEncoder, DepthDecoder
import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

from models.model_mde import ModelMDE

class DepthHints(ModelMDE):
    def __init__(self, model_path, device=None, **kwargs):
        self.encoder = ResnetEncoder(50, False)
        self.depth_decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        self.device = device
        #self.model, self.input_height, self.input_width = self.load(model_path, device)
        super(DepthHints, self).__init__(device=self.device, model_path=model_path, **kwargs)
    
    def load(
        self, model_path=None, device=None, **kwargs
    ):
        encoder = self.encoder
        depth_decoder = self.depth_decoder
        
        encoder_path = os.path.join(model_path, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()
        
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_decoder.to(device)
        depth_decoder.eval()
        
        model = lambda tensor_images: self.depth_decoder(
                self.encoder(tensor_images)
            )
        
        return model, feed_height, feed_width
    
    def predict(self, img, return_raw=False):
        #preprocess
        if not torch.is_tensor(img):
            img = T.ToTensor()(img).unsqueeze(0)
        
        H, W = int(img.shape[-2]), int(img.shape[-1])
        
        img_tensor = T.Resize((self.input_height, self.input_width), 
                       interpolation=T.InterpolationMode.BILINEAR)(img)
        img_tensor = img_tensor.to(self.device)
        
        #predict
        if return_raw:
            features = self.encoder(img_tensor)
            output = self.depth_decoder(features)
            
            return features, output
        
        disparity = self.model(img_tensor)[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disparity, (H, W), mode="bilinear", align_corners=False
        )
        
        return disp_resized
        
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
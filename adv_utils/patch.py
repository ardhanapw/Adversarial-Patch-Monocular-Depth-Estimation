import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import random

def create_mask(p):
    h, w = p.shape[-2], p.shape[-1]
    return torch.full((1, h, w), 1, dtype=torch.float32)

def apply_patch(augmented_patch: torch.Tensor,
                rgb: torch.Tensor,
                target_size:float,
                current_batch_size: int,
                bboxes: torch.Tensor,
                mask: torch.Tensor = None,
):
    
    #print("function")
    #print(rgb.shape)

    device = rgb.device
    #rgb = rgb.permute(0, 3, 1, 2)
    
    B, C, H, W = rgb.shape
    assert B == current_batch_size, f"{B=} != {current_batch_size=}"

    transformer = PatchTransformer().to(device)
    applier     = PatchApplier().to(device)

    if mask is None:
        mask = create_mask(augmented_patch).to(device)
    else:
        mask = mask.to(device)

    img_size = (W, H)
    #print("In apply patch")
    #print(target_size)
    patch_t, mask_t = transformer(
        patch=augmented_patch,
        mask=mask,
        batch_size=B,
        img_size=img_size,
        bboxes=bboxes,
        target_size=target_size,
        do_rotate=False,
        train=True,
    )

    final_images = applier(rgb, patch_t, mask_t)


    return final_images, mask_t


class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, patch, mask):
        #img_batch = img_batch.permute(0, 3, 2, 1)
        #print("apply patch")
        #print(img_batch.shape)
        #print(patch.shape)
        #print(mask.shape)
        #print(torch.max(img_batch))
        #print(torch.max(patch))
        #print(torch.max(mask))
        
        patched_img_batch = torch.mul((1 - mask), img_batch) + torch.mul(mask, patch)
        return patched_img_batch

class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.05
        self.max_brightness = 0.05
        self.noise_factor = 0.10
        self.minangle = 0#-20
        self.maxangle = 0#20
        self.minsize = 0.25#0.35
        self.maxsize = 0.25#0.45

        self.min_x_off = 0#-200
        self.max_x_off = 0#200
        self.min_y_off = 0#-80
        self.max_y_off = 0#80
        self.max_x_trans = 0#0.1
        self.min_x_trans = 0#-0.1
        self.max_y_trans = 0#0.1
        self.min_y_trans = 0#-0.1

    def normalize_transforms(self, transforms, W, H):
        theta = torch.zeros(transforms.shape[0], 2, 3).cuda()
        theta[:, 0, 0] = transforms[:, 0, 0]
        theta[:, 0, 1] = transforms[:, 0, 1]*H/W
        theta[:, 0, 2] = transforms[:, 0, 2]*2/W + theta[:, 0, 0] + theta[:, 0, 1] - 1

        theta[:, 1, 0] = transforms[:, 1, 0]*W/H
        theta[:, 1, 1] = transforms[:, 1, 1]
        theta[:, 1, 2] = transforms[:, 1, 2]*2/H + theta[:, 1, 0] + theta[:, 1, 1] - 1

        return theta
    
    def get_perspective_transform(self, src, dst):
        B = src.size(0)
        ones = torch.ones(B, 4, 1, device=src.device)
        zeros = torch.zeros_like(ones)

        x, y = src[:, :, 0:1], src[:, :, 1:2]
        u, v = dst[:, :, 0:1], dst[:, :, 1:2]

        A = torch.cat([
            torch.cat([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=2),
            torch.cat([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=2)
        ], dim=1)  # shape: (B, 8, 8)

        b = torch.cat([u, v], dim=1)  # shape: (B, 8, 1)

        h = torch.linalg.solve(A, b)  # shape: (B, 8, 1)
        h = h.view(B, 8)
        h = torch.cat([h, torch.ones(B, 1, device=src.device)], dim=1)
        return h.view(B, 3, 3)

    def forward(self, patch, mask, batch_size, img_size, bboxes, target_size, do_rotate=True, do_perspective=False, train=True):
        device = patch.device
        # Determine size of padding
        #print("Patch Transformer")
        #pad = (img_size - patch.size(-1)) / 2
        #print(img_size)
        img_width, img_height = img_size
        img_minimum_dim = min(img_width, img_height)
        #print(patch.shape, mask.shape)
        
        if patch.size(-1) > img_minimum_dim:
            patch = F.interpolate(
                patch.unsqueeze(0), size=(img_minimum_dim, img_minimum_dim), 
                mode="bilinear", align_corners=False
            )
            mask = F.interpolate(
                mask.unsqueeze(0), size=(img_minimum_dim, img_minimum_dim), 
                mode="nearest"  # nearest for masks, no interpolation blur
            )
        
        #patch is square
        width_diff = max(img_width - patch.size(-1), 0)
        height_diff = max(img_height - patch.size(-1), 0)
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top
        #pad_width = (img_width - patch.size(-1)) / 2
        #pad_height = (img_height - patch.size(-1)) / 2
        
        # Make a batch of patches
        adv_batch = patch.expand(batch_size, -1, -1, -1)
        mask_batch = mask.expand(batch_size, -1, -1, -1)

        if train:
            contrast = torch.empty_like(adv_batch).uniform_(self.min_contrast, self.max_contrast)
            brightness = torch.empty_like(adv_batch).uniform_(self.min_brightness, self.max_brightness)
            noise = torch.empty_like(adv_batch).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise transformation, and then clamp
            adv_batch = adv_batch * contrast + brightness + noise
            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Pad patch and mask to image dimensions
        # mypad = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)
        #mypad = nn.ConstantPad2d((160, 544, int(pad + 0.5), int(pad)), 0)
        #mypad = nn.ConstantPad2d((int(pad_width + 0.5), int(pad_width), int(pad_height + 0.5), int(pad_height)), 0)
        mypad = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)
        
        adv_batch = mypad(adv_batch)
        mask_batch = mypad(mask_batch)
        
        # Rotation and rescaling transforms
        if do_rotate:
            angle = torch.FloatTensor(1).uniform_(self.minangle, self.maxangle).to(device).expand(batch_size)
        else:
            angle = torch.zeros(batch_size, device=device)
        angle_rad = angle * math.pi / 180
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)
            
        center = torch.tensor([adv_batch.shape[3] / 2, adv_batch.shape[2] / 2], device=device).expand(batch_size, 2)

        rotation = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        scale = torch.ones(batch_size, device=device)
        # Translation
        translation = torch.eye(3, 3).cuda()
        translation = translation.expand(batch_size, -1, -1).clone()
        
        #resize to fit bounding box  
        x1 = (bboxes[:, 0] * img_width)
        y1 = (bboxes[:, 1] * img_height)
        x2 = (bboxes[:, 2] * img_width)
        y2 = (bboxes[:, 3] * img_height)
        
        #print("patch transformer too")
        #print(x1, y1, x2, y2)
        #print((y2-y1) * (x2-x1))
        
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        
        bbox_width = (x2 - x1).clamp(min=1.0)
        bbox_height = (y2 - y1).clamp(min=1.0)
        #print(bbox_width, bbox_height)
        #print(target_size)
        
        #maximum patch scaling possible in the bounding box
        if target_size is None or target_size == 'None':
            target_size = random.uniform(0, 1) #torch.rand(0, 1).to(device)

        side = torch.sqrt(target_size * bbox_width * bbox_height)
        max_side = torch.min(bbox_width, bbox_height)
        side = torch.min(side, max_side)
        scale = side/patch.size(-1)
        ratio = ((scale * patch.size(-1)) ** 2)/(bbox_width*bbox_height)
        #print("ratio")
        #print(ratio)
        #print(bbox_width.shape)
        #print(side)
        #print(target_size*bbox_width*bbox_height)

        
        x_off = x_center - (img_width / 2.0)
        y_off = y_center - (img_height / 2.0)
        translation[:, 0, 2] = x_off
        translation[:, 1, 2] = y_off
        
        #check ratio between resized patch and bbox

        
        #ensure placement is within bbox
        bbox_mask = torch.zeros((batch_size, 1, img_height, img_width), device=device, dtype=mask_batch.dtype)
        x_min = x1.floor().clamp(0, img_width - 1).int()
        y_min = y1.floor().clamp(0, img_height - 1).int()
        x_max = x2.ceil().clamp(1, img_width).int()
        y_max = y2.ceil().clamp(1, img_height).int()
        
        for b in range(batch_size):
            bbox_mask[b, 0, y_min[b]:y_max[b], x_min[b]:x_max[b]] = 1.0
        
        """
        # Resize
        current_patch_size = adv_batch.size(-2)
        if random_size:
            size = torch.FloatTensor(1).uniform_(self.minsize, self.maxsize).to(device)
            target_size = current_patch_size * size#(size ** 2)
        else:
            target_size = torch.tensor([current_patch_size * (0.4 ** 2)], device=device)
        scale = (target_size / current_patch_size).expand(batch_size)
        """

        # Rotate
        rotation[:, 0, 0] = cos * scale
        rotation[:, 0, 1] = -sin * scale
        rotation[:, 1, 0] = sin * scale
        rotation[:, 1, 1] = cos * scale
        rotation[:, 0, 2] = center[:, 0] * (1 - cos*scale) - center[:, 1] * (sin*scale)
        rotation[:, 1, 2] = center[:, 1] * (1 - cos*scale) + center[:, 0] * (sin*scale)

        """
        if rand_loc:
            x_off = torch.FloatTensor(1).uniform_(self.min_x_off, self.max_x_off).to(device) / scale
            y_off = torch.FloatTensor(1).uniform_(self.min_y_off, self.max_y_off).to(device) / scale
            translation[:, 0, 2] = x_off
            translation[:, 1, 2] = y_off
        """

        if do_perspective:
            # Perspective transform
            src_coord = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]], device=device).expand(batch_size, -1, -1)
            a_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            a_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            b_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            b_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            c_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            c_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            d_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            d_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            
            dst_coord = torch.tensor([[
                [0 + a_x, 0 + a_y],
                [1 + b_x, 0 + b_y],
                [0 + c_x, 1 + c_y],
                [1 + d_x, 1 + d_y],
            ]], device=device).expand(batch_size, -1, -1)


            perspective = self.get_perspective_transform(src_coord, dst_coord)
            M = rotation @ translation @ perspective
            
        else:
            M = translation @ rotation #rotation @ translation

        M_inv = torch.inverse(M)
        #theta = self.normalize_transforms(M_inv, W=512, H=256)
        #theta = self.normalize_transforms(M_inv, W=640, H=192)
        #theta = self.normalize_transforms(M_inv, W=1024, H=320)
        theta = self.normalize_transforms(M_inv, W=img_width, H=img_height)

        grid = F.affine_grid(theta, adv_batch.shape, align_corners=False)
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        mask_batch_t = F.grid_sample(mask_batch, grid, align_corners=False)
        mask_batch_t = (mask_batch_t > 0.5).float()
        
        #print("Patch transformer")
        #print((mask_batch_t > 0).sum(dim=(1,2,3)))
        if mask_batch_t.shape[1] != 1:
            bbox_mask = bbox_mask.expand(-1, mask_batch_t.shape[1], -1, -1)
        mask_batch_t = mask_batch_t * bbox_mask
        #print((mask_batch_t > 0).sum(dim=(1,2,3)))
        #print((bbox_mask > 0).sum(dim=(1,2,3)))
        
        #print(adv_batch_t.shape, (mask_batch_t > 0).sum(dim=(1,2,3)))

        return adv_batch_t, mask_batch_t

class PatchFunction(object):
    def __init__(self):
        super(PatchFunction, self).__init__()

    def LoadPatchFromImage(self, image_path, mask_path):
        noise_size = np.floor(self.image_size * np.sqrt(self.patch_size))
        patch_image = np.array(cv2.imread(image_path)).astype(np.float32)
        patch_image = self.resize_img(patch_image, (int(noise_size), int(noise_size)))/128. - 1
        patch = torch.FloatTensor(np.array([patch_image.transpose(2, 0, 1)]))
        mask_image = np.array(cv2.imread(mask_path)).astype(np.float32)
        mask_image = self.resize_img(mask_image, (int(noise_size), int(noise_size)))/256.
        mask = torch.FloatTensor(np.array([mask_image.transpose(2, 0, 1)]))
        return patch, mask

    def LoadPatchFromNpy(self, image_path, mask_path):
        patch_image = np.load(image_path).astype(np.float32)
        mask_image = np.load(mask_path).astype(np.float32)
        patch = torch.FloatTensor(patch_image)
        mask = torch.FloatTensor(mask_image)
        return patch, mask

    def InitSquarePatch(self, image_size, patch_size):
        noise_dim = patch_size
        patch = torch.rand((3, noise_dim, noise_dim))
        mask = torch.full((3, noise_dim, noise_dim), 1, dtype=torch.float32)
        return patch, mask

    def InitCirclePatch(self, image_size, patch_size):
        patch, _ = self.InitSquarePatch(image_size, patch_size)
        mask = self.CreateCircularMask(patch.shape[-2], patch.shape[-1]).astype('float32')
        mask = torch.FloatTensor(np.array([mask, mask, mask]))
        return patch, mask

    def CreateCircularMask(self, w, h):
        center = [int(w/2), int(h/2)]
        radius = min(center[0], center[1], w-center[0], h-center[1])-2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def resize_img(self, img, size):
        width, height = size
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

"""
class Adversarial_Patch(PatchFunction):
    def __init__(self, patch_type, batch_size, image_size, patch_size, train=True, printability_file=None):
        super(Adversarial_Patch, self).__init__()
        self.patch_type = patch_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_transformer = PatchTransformer()
        self.patch_applier = PatchApplier()

        if train:
            self.printfile = printability_file
            self.create_fake_disp = CreateFakeDisp()
            self.calculate_nps = NonPrintabilityScore(self.printfile, self.patch_size).cuda()
            self.calculate_tv = TotalVariation().cuda()

    def initialize_patch_and_mask(self):
        if self.patch_type == 'square':
            patch, mask = self.InitSquarePatch(self.image_size, self.patch_size)
        elif self.patch_type == 'circle':
            patch, mask = self.InitCirclePatch(self.image_size, self.patch_size)
        print('=> Use "{}" patch with shape {}'.format(self.patch_type, patch.shape))
        return patch, mask

    def load_patch_and_mask_from_file(self, patch_path, mask_path, npy=True):
        patch, mask = self.LoadPatchFromNpy(patch_path, mask_path) if npy else self.LoadPatchFromImage(patch_path, mask_path)
        print('=> Load patch with shape {} from \'{}\''.format(patch.shape, patch_path))
        return patch, mask
"""
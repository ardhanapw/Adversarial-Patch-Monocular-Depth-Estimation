import torch
from typing import Tuple
import torch.nn as nn

class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(triplet_scores_fpath, size), requires_grad=False
        ).cuda()

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, "r", encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack(
                [torch.full((p_h, p_w), r), torch.full((p_h, p_w), g), torch.full((p_h, p_w), b)]
            )
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)

class AdversarialLoss:
    def __init__(
        self,
        disp_loss_weight: float = 1.0,
        nps_loss_weight: float = 0.01,
        tv_loss_weight: float = 0.25,
        nps_triplet_scores_fpath: str = None,
        loss_function: str = "bce",
    ):
        self.disp_loss_weight = disp_loss_weight
        self.nps_loss_weight = nps_loss_weight
        self.tv_loss_weight = tv_loss_weight

        if loss_function == "bce":
            self.loss_function = torch.nn.functional.binary_cross_entropy
        elif loss_function == "mse":
            self.loss_function = torch.nn.functional.mse_loss

        self.mse_loss = torch.nn.functional.mse_loss
        self.mae_loss = torch.nn.functional.l1_loss
        self.nps_triplet_scores_fpath = nps_triplet_scores_fpath

    def tv_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        adv_patch_dx = adv_patch - torch.roll(adv_patch, shifts=1, dims=1)
        adv_patch_dy = adv_patch - torch.roll(adv_patch, shifts=1, dims=2)

        loss = self.mae_loss(adv_patch_dx, torch.zeros_like(adv_patch_dx)) + self.mae_loss(
            adv_patch_dy, torch.zeros_like(adv_patch_dy)
        )

        return loss * self.tv_loss_weight

    def disp_loss(
        self,
        predicted_disparities: torch.Tensor,
        target_disparities: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.loss_function(
            predicted_disparities, target_disparities, reduction="none"
        )

        valid_indices = masks > 0.5
        valid_loss = loss[valid_indices]
        final_loss = torch.mean(valid_loss)

        return final_loss * self.disp_loss_weight

    def nps_loss(self, adv_patch: torch.Tensor) -> torch.Tensor:
        h, w = adv_patch.shape[-2], adv_patch.shape[-1]
        nps = NPSLoss(self.nps_triplet_scores_fpath, (h, w))
        nps_loss = nps(adv_patch)

        return nps_loss * self.nps_loss_weight

    def __call__(self, adv_patch, masks, predicted_disparities, target_disparities):
        # disp loss
        if self.disp_loss_weight > 0:
            disp_loss = self.disp_loss(
                predicted_disparities, target_disparities, masks
            )
        else:
            disp_loss = 0.0
            
        #nps loss
        if self.nps_loss_weight > 0:
            nps_loss = self.nps_loss(adv_patch)
        else:
            nps_loss = 0.0

        # TV loss
        if self.tv_loss_weight > 0:
            tv_loss = self.tv_loss(adv_patch)
        else:
            tv_loss = 0.0

        # Compute the total loss
        total_loss = disp_loss + nps_loss + tv_loss

        return {
            "total_loss": total_loss,
            "disp_loss": disp_loss,
            "nps_loss": nps_loss,
            "tv_loss": tv_loss,
        }
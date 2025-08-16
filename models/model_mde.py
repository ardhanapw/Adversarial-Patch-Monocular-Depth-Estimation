import torch
from abc import ABC, abstractmethod
from typing import Tuple, Callable

class ModelMDE(ABC):
    def __init__(self, device=None, **kwargs):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.input_height, self.input_width = self.load(
            device=self.device, **kwargs
        )
        
    def __call__(self, img_batch_tensor, **kwargs):
        prediction = self.predict(img_batch_tensor)
        return prediction
    
    @abstractmethod
    def load(
        self, model_path=None, device=None, **kwargs
    ) -> Tuple[Callable, int, int]:
        pass
    
    @abstractmethod
    def predict(self, tensor_images, **kwargs):
        pass
    
    @abstractmethod
    def plot(self, image, prediction, save=False, save_path=None):
        pass
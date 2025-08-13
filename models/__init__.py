from models.monodepth2.model import MonoDepth2
from models.depth_hints.model import DepthHints

def load_models(model_name, model_path, device):
    if model_name == 'monodepth2':
        return MonoDepth2(model_path, device)
    elif model_name == 'depth_hints':
        return DepthHints(model_path, device)
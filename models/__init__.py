from models.monodepth2.model import MonoDepth2


def load_models(model_name, model_path, device):
    if model_name == 'monodepth2':
        return MonoDepth2(model_path, device)
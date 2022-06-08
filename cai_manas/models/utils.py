import os

def get_local_ckpt(model_name):
    """Convert the name of the model in the CAI data registry to a local checkpoint path.

    Args:
        model_name (:obj:`string`):
            The model name in the CAI data registry. If it starts with 'model_archive', it is assumed to be a path
            within the data registry. Otherwise, it is assumed to be a champion model inside the champion_models
            directory.

    Returns:
        The local directory name you can feed to AutoModel.from_pretrained.
    """

    data_base_path = os.environ['CAI_DATA_BASE_PATH']
    if not model_name.startswith('model_archive'):
        model_name = os.path.join('champion_models', model_name)
    return os.path.join(data_base_path, model_name)

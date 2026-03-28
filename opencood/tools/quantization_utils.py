import opencood.hypes_yaml.yaml_utils as yaml_utils

def load_quantization_yaml(path, hypes):
    """
    Loads the quantization hypes located in 'path', then merge the resulting dictionnary with 'hypes'.
    """

    q_hypes = yaml_utils.load_yaml(path)

    model_args = hypes['model']['args']
    for k, v in q_hypes.items():
        # check for no identical keys

        if model_args[k].keys() & v.keys():
            common_keys = model_args[k].keys() & v.keys()
            raise Exception("Quantization hype and model hype both have the following key(s): {}".format(", ".join(common_keys)))

        model_args[k] |= q_hypes[k]

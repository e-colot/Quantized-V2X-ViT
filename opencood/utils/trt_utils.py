import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils


def _parser():
    parser = argparse.ArgumentParser(description="Model selector")
    parser.add_argument('--model', type=str,
                        required=True)
    parser.add_argument('--type', type=str, default='torchscript')
    opt = parser.parse_args()
    return opt

class _Arguments:
    def __init__(self, modelName):
        print('Default parameters used')
        self.model_name = modelName
        self.global_sort_detections = False
        self.fusion_method = 'intermediate'
        if modelName == "v2xvit":
            self.model_dir = 'opencood/logs/v2x-vit'
        elif modelName == "ppif":
            self.model_dir = 'opencood/logs/pointPillarIntermediateFusion'

        assert self.fusion_method in ['late', 'early', 'intermediate']


def load_params(parser_opt=None):
    if parser_opt is None:
        parser_opt = _parser()

    valid_model_names = {
        "v2xvit",
        "ppif" # point pillar intermediate fusion
    }
    if parser_opt.model not in valid_model_names:
        raise ValueError(f"Invalid TRT_STAGE={parser_opt.model}. Use one of {sorted(valid_model_names)}")
    
    valid_compiler_type = {
        "torchscript",
        "onnx"
    }
    if parser_opt.type not in valid_compiler_type:
        raise ValueError(f"Invalid TRT_STAGE={parser_opt.type}. Use one of {sorted(valid_compiler_type)}")
    
    opt = _Arguments(parser_opt.model)

    hypes = yaml_utils.load_yaml(None, opt)

    # for convenient shape logs naming
    hypes['name'] = parser_opt.model

    return hypes, opt, parser_opt



## Setup
Please refer to the [original OpenCOOD repository](https://github.com/DerrickXuNu/OpenCOOD) for a full explanation about data downloading, libraries setup and so on.

## Basic commands

Here are a few commands that can be run to check whether everything works fine

### Data sequence visualization
The `validate_dir` in `opencood/hypes_yaml/visualization.yaml` should match the validation dataset of your choice.

```python
python opencood/visualization/vis_data_sequence.py
```


### Train your model
To train the V2X-ViT model, run the following command

```python
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_v2xvit.yaml
```

### Test the model
The `validation_dir` in `config.yaml` under your checkpoint folder (here `opencood/v2x-vit/`)
should match the testing dataset path.

```python
python opencood/tools/inference.py --model_dir opencood/v2x-vit/ --fusion_method intermediate
```

The evaluation results  will be dumped in the model directory. 

# Finetuning DINOv2 for downstream task

This repository is for training DINOv2 for downstream tasks.<br>
**It is not for self-supervised learning.**

## Table of Contents

- [Installation](#installation)
  - [Using `requirements.txt`](#using-requirements.txt)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## Supproted Features

- Data Pararell
- Class Balanced Loss
- Rare Class Sampling
- Select optimizer
- Freeze/Unfreeze backbone

## Installation

Install the required packages using `requirements.txt`.
Because of xformer, it requires the latest version of pytorch. However you can use different version of xformer and pytorch.

### Using `requirements.txt`

```bash
pip install -r requirements.txt
```

### Dataset Prearation

The script requires a dataset formatted as below.

```
Data
├── ...
├── Class4
│   ├── Img1.png
│   ├── Img2.png
│   ├── ...
├── Class5
│   ├── Img1.png
│   ├── Img2.png
│   ├── ...
├── ...
```

**Data preprocessing**: Please run the following scripts to generate the `class_stats.json`.

```bash
python tools/preprocess.py /path/to/yout/dataset
```

## Training

You can launch the training code by using:

```bash
bash train.sh
```

You can set your training arguments at [config.py](./configs/config.py).<br>
There is a setting for Rare Class Sampling(RCS). It is a setting for long-talied class motivated from [DAFormer](https://github.com/lhoyer/DAFormer)
This will sample the rare class more often during the iteration. However it has a risk of model to not see some classes.
It is more suitable for **multi-class classifiaction**.

<details>
<summary>Training arguments</summary>

- `batch_per_gpu` (int): Number of samples per GPU in each forward step (default: 16).
- `num_gpu` (int): Number of GPUs used for training (default: 1).
- `resize` (tuple): The size to which input images are resized (default: (224, 224)).
- `mean` (list): Mean normalization values for each channel in RGB format (default: [0.485, 0.456, 0.406]).
- `std` (list): Standard deviation normalization values for each channel in RGB format (default: [0.229, 0.224, 0.225]).
- `optimizer` (dict): Optimizer settings.
  - `type`: Optimizer type (default: 'SGD').
  - `params`: Additional optimizer parameters, such as momentum (default: 0.9).
  - `learning_rate`: Learning rates for different parts of the model.
    - `head_lr`: Learning rate for the head (default: 1e-3).
    - `backbone_lr`: Learning rate for the backbone (default: 1e-6).
- `scheduler` (dict): Learning rate scheduler settings.
  - `type`: Scheduler type (default: 'linear').
  - `params`: Additional scheduler parameters like warmup ratio (default: 0.03).
- `do_eval` (bool): Whether to perform evaluation during training (default: False).
- `num_train_epoch` (int): Number of epochs for training (default: 100).
- `model` (dict): Model architecture settings.
  - `backbone`: Backbone model type (default: 'dinov2_l').
  - `head`: Classification head type (default: 'single').
  - `num_classes`: Number of output classes (default: 3).
  - `freeze_backbone`: Whether to freeze the backbone during training (default: False).
- `loss` (dict): Loss function settings.
  - `loss_type`: Type of loss function (default: 'CE_loss').
  - `beta`: Beta parameter for class-balanced loss (default: None).
  - `gamma`: Gamma parameter for focal loss (default: None).
- `dataset` (dict): Dataset paths.
  - `train`: Training dataset settings.
    - `data_root`: Root directory of the training dataset.
  - `eval`: Evaluation dataset settings.
    - `data_root`: Root directory of the evaluation dataset.
- `max_checkpoint` (int): Maximum number of checkpoints to keep (default: 1).

**Note:** The backbone learning rate is often set to be much smaller than the head learning rate to prevent overfitting the pretrained layers.

</details>

## Evalutation

You can evaluate your model by using:

```bash
bash eval.sh
```

The evaluation will calculate the top-k accuracy together.

## TODO

- [ ] Multi-label classification
- [ ] Segmentation

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Dino-v2-Finetuning,
  author = {Yuwon Lee},
  title = {Dino-V2-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/DINOv2-Finetune}
}
```

## Acknowledgement

This project is based on

- [DINOv2](https://github.com/facebookresearch/dinov2)

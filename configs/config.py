class Config:
    def __init__(self):
        self.batch_per_gpu = 16
        self.num_gpu = 1
        self.resize = (224,224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.optimizer = {
            'type': 'SGD',
            'params': {
                'momentum': 0.9
            },
            'learning_rate': {
                'head_lr': 1e-3,
                'backbone_lr': 1e-6
            }
        }
        self.scheduler = {
            'type': 'linear',
            'params': {
                'warmup_ratio': 0.03
            }
        }
        self.do_eval = False
        # self.eval_step = 2000 # This is only used when the training is iteration based (not epoch based)
        # self.iterations = "8k"  # This is only used when the training is iteration based (not epoch based)
        self.num_train_epoch = 100
        self.model = {
            'backbone': 'dinov2_l', # 'dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g', 'siglip_384'
            'head': 'single', # 'single', 'mlp'
            # 'hidden_dims': [512, 256], # This is only used when head is 'mlp'
            'num_classes': 3,
            'freeze_backbone': False
        }
        self.loss = {
            'loss_type': 'CE_loss', # 'CE_loss', 'class_balanced_CE_loss', 'Focal_loss', 'class_balanced_Focal_loss'
            # 'beta': 0.99,
            # 'gamma': 0.5
        }
        self.dataset = {
            'train': {
                'data_root': '/path/to/your/dataset',
                # The rare_class_sampling is used when it is iteration based.
                # This makes the training to sample the rare classes more frequently. However, it has a risk of not seeing the all data.
                # rare_class_sampling might be useful when your class is multi-label classification.
                # 'rare_class_sampling': {
                #     'class_temp': 0.1
                # }
            },
            'eval': {
                'data_root': '/path/to/your/dataset',
            }
        }
        self.max_checkpoint = 1

    def get_cfg(self):
        return vars(self)
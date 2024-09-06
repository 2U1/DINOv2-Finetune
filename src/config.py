class Config:
    def __init__(self):
        self.batch_per_gpu = 64
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
        self.eval_step = 2000
        # self.iterations = "8k"  # 예: 8,000 iterations
        self.num_train_epoch = 100
        self.model = {
            'backbone': 'dinov2_l',
            'head': 'single',
            'num_classes': 713,
            'freeze_backbone': False
        }
        self.loss = {
            'loss_type': 'class_balanced_CE_loss', # 'CE_loss', 'class_balanced_CE_loss', 'Focal_loss', 'class_balanced_Focal_loss'
            'beta': 0.99,
            'gamma': 0.5
        }
        self.dataset = {
            'train': {
                'data_root': '/home/workspace/reid/train',
                'rare_class_sampling': {
                    'class_temp': 0.1
                }
            },
            'eval': {
                'data_root': '/home/workspace/reid/test',
            }
        }
        self.output_dir = './output/0906_CBCELoss'
        self.max_checkpoint = 10

    def get_cfg(self):
        return vars(self)  # Config 객체를 딕셔너리로 반환
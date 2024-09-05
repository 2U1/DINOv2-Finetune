class Config:
    def __init__(self):
        # 기본 설정값 초기화
        self.batch_per_gpu = 64
        self.resize = (224,224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.optimizer = {
            'type': 'AdamW',
            'params': {
                'weight_decay': 0.01,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8
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
        self.num_train_epoch = 3
        self.model = {
            'backbone': 'dinov2_l',
            'head': 'mlp',
            'hidden_dims': [512, 256],
            'dropout': 0.3,
            'num_classes': 713
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
        self.output_dir = './output/0905_mlp'

    def get_cfg(self):
        return vars(self)  # Config 객체를 딕셔너리로 반환
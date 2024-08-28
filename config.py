class Config:
    def __init__(self):
        # 기본 설정값 초기화
        self.batch_size = 512
        self.resize = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.scheduler_type = "linear"
        self.warmup_ratio = 0.03
        self.eval_step = 500
        self.iterations = "8k"  # 예: 8,000 iterations
        self.num_classes = 10
        self.backbone = 'dinov2_l'
        self.head = 'linear'
        self.dataset = {
            'train': {
                'data_root': 'path/to/train_data',
                'rare_class_sampling': {
                    'class_temp': 0.1
                }
            },
            'eval': {
                'data_root': 'path/to/eval_data',
            }
        }
        self.output_dir = './output'

    def get_cfg(self):
        return vars(self)  # Config 객체를 딕셔너리로 반환
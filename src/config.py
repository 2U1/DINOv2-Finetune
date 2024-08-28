class Config:
    def __init__(self):
        # 기본 설정값 초기화
        self.batch_per_gpu = 64
        self.resize = (224,224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.scheduler_type = "linear"
        self.warmup_ratio = 0.03
        self.eval_step = 500
        self.iterations = "8k"  # 예: 8,000 iterations
        self.num_classes = 713
        self.backbone = 'dinov2_l'
        self.backbone_lr = 1e-5
        self.head_lr = 1e-3
        self.head = 'linear'
        self.dataset = {
            'train': {
                'data_root': '/home/workspace/reid/train',
                'rare_class_sampling': {
                    'class_temp': 0.6
                }
            },
            'eval': {
                'data_root': '/home/workspace/reid/test',
            }
        }
        self.output_dir = './output/0828'

    def get_cfg(self):
        return vars(self)  # Config 객체를 딕셔너리로 반환
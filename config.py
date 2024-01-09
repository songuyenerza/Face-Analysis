from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.num_classes = 6
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0
config.fp16 = True
config.weight_decay = 5e-4
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.lr_name = 'cosine'
config.warmup_lr = 5e-7
config.min_lr = 5e-6

config.rec = "../../../data/face_cropped/images"
config.num_epoch = 200
config.warmup_epoch = 0
config.val_targets = []


config.save_path = './model_age_v2'
config.log = 'train_v2.txt'


config.warmup_step = 1000
config.total_step = 20000
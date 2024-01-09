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
config.optimizer = "sgd"
config.lr = 0.05
config.verbose = 2000
config.dali = False

config.rec = "../../../data/face_cropped/images"
config.num_epoch = 100
config.warmup_epoch = 0
config.val_targets = []


config.save_path = './model_age_v1'
config.log = 'train.txt'
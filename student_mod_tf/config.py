import tensorflow as tf

DATA_DIR_PATH_CELEBA = 'gs://kds-e6507e4369af642c1a0e9d3e31cf5b6dc5c6cb872c423e15dc4c348b'
DATA_DIR_PATH_FER = "gs://kds-f0d47e77ed743cd131b8a9fa7e57468f53dd823b732ee4645eb14527"

train_shard_suffix = 'data_*-9.tfrec'

train_set_path = sorted(tf.io.gfile.glob(DATA_DIR_PATH_CELEBA + f'/{train_shard_suffix}'))
REPLICAS = 8
batch_size = 128 * REPLICAS
image_shape = (256, 256)

train_set_len = 202599 
train_step = -(-train_set_len // batch_size)

train_shard_suffix = 'train_*-3.tfrec'
train_set_path_fer = sorted(tf.io.gfile.glob(DATA_DIR_PATH_FER + f'/train/{train_shard_suffix}'))

batch_size_fer = 32 * REPLICAS
val_set_len = 28662
val_step = -(-val_set_len // batch_size_fer)

start_lr = 0.001
min_lr = 0.0005
max_lr = 0.005
rampup_epochs = 10
sustain_epochs = 0
exp_decay = .8


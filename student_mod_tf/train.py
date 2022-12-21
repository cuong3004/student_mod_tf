import warnings
import os
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger
from wandb.keras import WandbCallback
from student_mod_tf.config import *
from student_mod_tf.callback import *
from student_mod_tf.model import MobileViT_XS
import tensorflow_addons as tfa
from student_mod_tf.model_bt import BarlowModel
from student_mod_tf.dataset import train_dataset_encode, \
                                train_dataset_fer_encode, \
                                train_step, \
                                val_step
                                

MIXED_PRECISION = True
XLA_ACCELERATE = True

try:  # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  
    strategy = tf.distribute.TPUStrategy(tpu)
    DEVICE = 'TPU'
except ValueError:  # detect GPUs
    strategy = tf.distribute.get_strategy() 
    DEVICE = 'GPU'
    
if DEVICE == "GPU":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    try: 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except: # Invalid device or cannot modify virtual devices once initialized.
        pass 
    
if MIXED_PRECISION:
    dtype = 'mixed_bfloat16' if DEVICE == "TPU" else 'mixed_float16'
    tf.keras.mixed_precision.set_global_policy(dtype)
    dtype_model = tf.bfloat16
    print('Mixed precision enabled')
else:
    dtype_model = tf.float32


if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')
    
AUTO  = tf.data.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync

print('REPLICAS           : ', REPLICAS)
print('TensorFlow Version : ', tf.__version__)
print('Eager Mode Status  : ', tf.executing_eagerly())
print('TF Cuda Built Test : ', tf.test.is_built_with_cuda)
print(
    'TF Device Detected : ', 
    'Running on TPU' if DEVICE == "TPU" else tf.test.gpu_device_name()
)

try:
    print('TF System Cuda V.  : ', tf.sysconfig.get_build_info()["cuda_version"])
    print('TF System CudNN V. : ', tf.sysconfig.get_build_info()["cudnn_version"])
except:
    pass



run = wandb.init(project="test_bt", name="mobilevit_se_5e-3")

callbacks = [cp_callback, 
             lr_callback, 
             WandbMetricsLogger(),
             gcs_callback
            ]


# Contrastive pretraining
with strategy.scope():
    pretraining_model = BarlowModel()
    
    pretraining_model.compile(
        main_optimizer=tfa.optimizers.LAMB(),
        probe_optimizer=keras.optimizers.Adam(),
    )
    pretraining_model.build(input_shape = (*image_shape,3))
    
#     pretraining_model.load_weights(path_checkpoint)


pretraining_history = pretraining_model.fit(
    train_dataset_encode, epochs=500,
    # initial_epoch=100,
    steps_per_epoch=train_step,
    # steps_per_epoch=1,
    validation_data=train_dataset_fer_encode, 
    validation_steps=val_step,
    callbacks=callbacks,
)

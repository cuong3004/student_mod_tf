from tensorflow import keras
from student_mod_tf.config import *
import wandb
from natsort import natsorted
import os 

class GCSCallback(keras.callbacks.Callback):

    def __init__(self):
        super(GCSCallback, self).__init__()
        self.name_old = ""
        
    def upload_file_to_gcs(self, src_path: str):
        
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(src_path, 'model.hdf5')
        wandb.log_artifact(artifact)

    def on_epoch_end(self, epoch, logs=None):
        
        try:
            list_dir = natsorted(os.listdir('/tmp/checkpoints'), reverse=True)
            cp_file = '/tmp/checkpoints/' + list_dir[0]
            src_path = os.path.join('/tmp/checkpoints', cp_file)
            if src_path != self.name_old:
                self.name_old = src_path
                self.upload_file_to_gcs(src_path=src_path)
                print(f"Epoch {str(epoch+1).zfill(5)}: Uploaded {src_path}\n")
        except:
            print("Dont Have file")



def lrfn(epoch):
  if epoch < rampup_epochs:
    return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
  elif epoch < rampup_epochs + sustain_epochs:
    return max_lr
  else:
    return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

lr_callback = keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

cp_callback = keras.callbacks.ModelCheckpoint(filepath='/tmp/checkpoints/model.{epoch:02d}-{loss:.2f}.hdf5',
                             monitor='loss',
                             save_freq='epoch',
                             verbose=1,
                             period=20,
                             save_best_only=True,
                             save_weights_only=True)

gcs_callback = GCSCallback()
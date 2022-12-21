from tensorflow import keras
import tensorflow as tf
from student_mod_tf.config import *
from student_mod_tf.transforms import bt_augmentor

AUTO = tf.data.AUTOTUNE


def deserialization_fn(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
        })
    
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, image_shape)
    image_1 = bt_augmentor(image)
    image_2 = bt_augmentor(image)

    image_1 = tf.keras.applications.mobilenet_v2.preprocess_input(image_1)
    image_2 = tf.keras.applications.mobilenet_v2.preprocess_input(image_2)
    # label = tf.cast(features['image/class/label'], tf.int64) - 1  # [0-999]
    
    return image_1, image_2

def tfrecords_loader(files_path, shuffle=False):
    datasets = tf.data.Dataset.from_tensor_slices(files_path)
    datasets = datasets.shuffle(len(files_path)) if shuffle else datasets
    datasets = datasets.flat_map(tf.data.TFRecordDataset)
    datasets = datasets.map(deserialization_fn, num_parallel_calls=AUTO)
    return datasets

def tfrecords_loader_fer(files_path, shuffle=False):
    datasets = tf.data.Dataset.from_tensor_slices(files_path)
    datasets = datasets.shuffle(len(files_path)) if shuffle else datasets
    datasets = datasets.flat_map(tf.data.TFRecordDataset)
    
    def deserialization_fn_fer(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            })

        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.resize(image, image_shape)
#         image_1 = bt_augmentor(image)
#         image_2 = bt_augmentor(image)

#         image_1 = tf.keras.applications.mobilenet_v2.preprocess_input(image_1)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        label = tf.cast(features['label'], tf.int64)  # [0-999]

        return image, label
    datasets = datasets.map(deserialization_fn_fer, num_parallel_calls=AUTO)
    return datasets


train_datasets = tfrecords_loader(train_set_path)
train_datasets_fer = tfrecords_loader_fer(train_set_path_fer)

train_dataset_encode = train_datasets.repeat().batch(batch_size).prefetch(AUTO)
batch_size_fer = 32 * REPLICAS
train_dataset_fer_encode = train_datasets_fer.repeat().batch(batch_size_fer).prefetch(AUTO)


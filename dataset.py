import tensorflow as tf
import numpy as np
import math
from pathlib import Path

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(latent):
    feature = {
        'latent' : _bytes_feature(latent)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def generate_tfrecords(latent_size,save_folder):
    
    tf_folder = Path(save_folder)
    tf_folder.mkdir(parents=True, exist_ok=True)

    num_samples = 50000
    num_images_pshard = 500
    num_tfrecords=math.ceil(num_samples/num_images_pshard)

    sample_count=0

    for i in range(num_tfrecords):
        print('Processing of tf record number {} out of {}'.format(i+1,num_tfrecords))
        tf_path = tf_folder.joinpath('data_train_shard{}.tfrec'.format(i))
        with tf.io.TFRecordWriter(str(tf_path)) as tf_record_writer : 
            for sample in range(num_images_pshard):
                latent = np.array(tf.random.normal((1, latent_size))).astype('float32')
                latent = latent.ravel().tostring()
                tf_record_writer.write(serialize_example(latent))
                sample_count +=1
                print('{} / {} samples processed'.format(sample_count,num_samples))

def parse_image(serialized):
    features = {
        'latent': tf.io.FixedLenFeature([], tf.string),  # image was converted to string
    }

    # getting the data
    data = tf.io.parse_single_example(serialized=serialized, features=features) 
    # getting the image in right format
    latent = data['latent']
    latent = tf.io.decode_raw(latent, tf.float32)
    return latent

def get_tf_dataset(tf_folder):

    tffolder = Path(tf_folder)
    num_parallel_calls = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.list_files(filenames=tffolder.joinpath('data_train_shard*.tfrec'),shuffle=False)
    cycle_length = 1 if num_parallel_calls is None else num_parallel_calls

    dataset = dataset.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(x, compression_type=None),
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
    )
    # extracting data
    dataset = dataset.map(lambda x: parse_image(x),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def batch_dataset(dataset,batch_size):
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset 
import tensorflow as tf
from pathlib import Path

def serialize_example(data,shape):
    feature = {
        'img' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
        'shape' : tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_image(serialized):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),  # image was converted to string
        'shape': tf.io.FixedLenFeature([4], tf.int64)  # 4 because we added channels
    }

    # getting the data
    data = tf.io.parse_single_example(serialized=serialized, features=features) 

    # getting the image in right format
    image = data['image']
    image = tf.io.decode_raw(image, tf.float32)
    image = tf.reshape(image,data['shape']) # reshape after deserialization

    return image

def get_dataset(data,batch_size):

    dataset = tf.data.Dataset.from_tensor_slices(data)
    #dataset = dataset.map(lambda x: parse_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_tf_dataset(tf_folder,batch_size):

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
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
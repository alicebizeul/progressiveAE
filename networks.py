import tensorflow as tf
from pathlib import Path
import numpy as np
import dataset


class Encoder:

    def __init__(self, latent_size):
        super(Encoder, self).__init__()

        # static parameters 
        self.latent_size = latent_size
        self.num_channels = 1
        self.dimensionality = 3
        self.fmap_base = 512
        self.fmap_max = 8192

        # dynamic parameters
        self.current_resolution = 1
        self.current_width = 2 ** self.current_resolution
        self.growing_encoder = self.make_Ebase(nf=self._nf(1))
        self.train_encoder = self.growing_encoder
    
    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def make_Ebase(self,nf):

        images = tf.keras.layers.Input(shape= (2,)*self.dimensionality + (nf,), name='images_2iso')

        # Final dense layer
        x = tf.keras.layers.Flatten()(images)
        x = tf.keras.layers.Dense(self.latent_size*2)(x)
        # ADD ACTIVATION AND DENSE ??? 

        # Getting Normal distribution parameters
        mu = x[:self.latent_size]
        sigma = tf.nn.softplus(x[self.latent_size:])

        # Latent code - computed for loss evaluation
        #z = tensorflow_probability.distributions.Normal(loc=mu, scale=sigma)

        return tf.keras.models.Model(inputs=[images], outputs=[mu,sigma], name='mu_sigma')

    def make_Eblock(self,name,nf):
        print(nf)

        # on fait cette approche car on ne sait pas la taille donc on met pas un input
        block_layers = []

        block_layers.append(tf.keras.layers.Convolution3D(nf, kernel_size=3, strides=1, padding='same'))
        block_layers.append(tf.keras.layers.Activation(tf.nn.leaky_relu))

        block_layers.append(tf.keras.layers.Convolution3D(nf, kernel_size=3, strides=2, padding='same')) # check padding
        block_layers.append(tf.keras.layers.Activation(tf.nn.leaky_relu))

        return tf.keras.models.Sequential(block_layers, name=name)

    def _nf(self, stage): 
        # computes number of filters for each layer
        return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

    def _weighted_sum(self):
        return tf.keras.layers.Lambda(lambda inputs : (1-inputs[2])*inputs[0] + (inputs[2])*inputs[1])

    def add_resolution(self):
        
        # Add resolution
        self.update_res()

        # Gan images
        images = tf.keras.layers.Input(shape=(self.current_width,)*self.dimensionality+ (self.num_channels,),name = 'GAN_images')

        # Compression block
        name = 'block_{}'.format(self.current_resolution)
        e_block = self.make_Eblock(name=name,nf=self._nf(self.current_resolution-1))

        # Channel compression
        from_rgb_1 = tf.keras.layers.AveragePooling3D()(images)
        from_rgb_1 = tf.keras.layers.Conv3D(self._nf(self.current_resolution-1), kernel_size=1, padding='same', name='from_rgb_1')(from_rgb_1)

        print(self._nf(self.current_resolution))
        from_rgb_2 = tf.keras.layers.Conv3D(self._nf(self.current_resolution), kernel_size=1, padding='same', name='from_rgb_2')(images)
        from_rgb_2 = e_block(from_rgb_2)

        lerp_input = self._weighted_sum()([from_rgb_1, from_rgb_2, tf.constant(2,dtype=tf.float32)]) # RANDOM ALPHA

        # Getting latent code 
        #block_output = e_block(lerp_input)
        e_output = self.growing_encoder(lerp_input)

        # Updating the model
        self.growing_encoder = tf.keras.Sequential([e_block,self.growing_encoder]) # without channel compression
        self.train_encoder = tf.keras.Model(inputs=[images],outputs=[e_output]) # with channel compression
        print(self.train_encoder.summary())
      
class Decoder(): 

    def __init__(self, latent_size, generator_folder):
        super(Decoder, self).__init__()
        
        # static parameters
        self.latent_size = latent_size
        self.model_folder = generator_folder

        # dynamic parameters
        self.current_resolution = 1
        self.current_width = 2** self.current_resolution
        self.decoder = None

    def get_model(self,folder,res):
        # find the model for the appropriate resolution
        path = Path(folder)
        return str(list(path.glob('g_{}.h5'.format(res)))[0].resolve()) # check if single one

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def add_resolution(self):
        self.update_res()
        self.decoder = tf.keras.models.load_model(self.get_model(self.model_folder,self.current_resolution), custom_objects={'leaky_relu': tf.nn.leaky_relu}, compile=True)
        self.decoder.trainable = False

    def get_decoder(self):
        return self.decoder

    def get_currentres(self):
        return self.current_resolution

class Generator():

    def __init__(self, latent_size, generator_folder):
        super(Generator,self).__init__()

        #static parameters
        self.latent_size = latent_size  # LOOK UP WHAT SIZE
        self.model_folder = generator_folder

        # dynamic 
        self.current_resolution = 1
        self.current_width = 2**3
        self.generator = None

    def get_model(self,folder,res):
        # find the model for the appropriate resolution
        path = Path(folder)
        print([x for x in path.iterdir()])
        print(list(path.glob('**/g_3.h5')))
        prtin('**/g_{}.h5'.format(res))
        print(list(path.glob('**/g_{}.h5'.format(res))))
        try:
            print(str(list(path.glob('g_{}.h5'.format(res)))[0].resolve()))
            return str(list(path.glob('**/g_{}.h5'.format(res)))[0].resolve())
        except: print('No pretrained model for this resolution')

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def add_resolution(self):
        self.update_res()
        self.generator = tf.keras.models.load_model(self.get_model(self.model_folder,self.current_resolution), custom_objects={'leaky_relu': tf.nn.leaky_relu}, compile=True)
        self.generator.trainable = False

    def generate_latents(self,num_samples):
        # TO DO : instead of storing images, use latents, no need to store in TFRecords
        latents = []
        for i in range(num_samples):
            latent = tf.random.normal((1, self.latent_size))
            latents.append(latent)
        return latents

    def generate_samples(self,num_tfrecords,save_folder):
        
        tf_folder = Path(save_folder)
        num_images_pshard = 200

        for i in range(num_tfrecords):
            print('Processing of tf record number {} out of {}'.format(i+1,num_tfrecords))
            tf_path = tf_folder.joinpath('data_train_shard{}.tfrec'.format(i))
            with tf.io.TFRecordWriter(tf_path) as tf_record_writer : 
                for samples in range(num_images_pshard):

                    latents = tf.random.normal((1, self.latent_size)) 
                    images = self.generator(latents)
                    img_data = np.squeeze(np.array(images[0])).astype('float32') # remove single dimension
                    img_data = img_data.ravel().tostring()
                    img_shape = img_data.shape

                    if len(img_shape)==3: # channels
                        img_shape = np.append(img_shape, 1)

                    tf_record_writer.write(dataset.serialize_example(img_data, img_shape))
        


            


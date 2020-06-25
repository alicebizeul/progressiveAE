import tensorflow as tf
from pathlib import Path
import numpy as np
import dataset
import tensorflow_probability as tfp

tfd = tfp.distributions

class Encoder:

    def __init__(self, latent_size):
        super(Encoder, self).__init__()

        # static parameters 
        self.latent_size = latent_size
        self.num_channels = 1
        self.dimensionality = 3
        self.fmap_base = 2048
        self.fmap_max = 8192
    
        # dynamic parameters
        self.current_resolution = 1
        self.current_width = 2 ** self.current_resolution
        self.growing_encoder = self.make_Ebase(nf=self._nf(1))
        self.train_encoder = tf.keras.Sequential(self.growing_encoder,name='sequential')
    
    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def make_Ebase(self,nf):

        images = tf.keras.layers.Input(shape= (2,)*self.dimensionality + (nf,), name='images_2iso')

        # Final dense layer
        x = tf.keras.layers.Flatten()(images)
        x = tf.keras.layers.Dense(2*self.latent_size,activation=None)(x)
        return tf.keras.models.Model(inputs=[images], outputs=[x], name='z_params')

    def make_Eblock(self,name,nf):

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
        alpha = tf.keras.layers.Input(shape=[], name='e_alpha')

        # Compression block
        name = 'block_{}'.format(self.current_resolution)
        e_block = self.make_Eblock(name=name,nf=self._nf(self.current_resolution-1))

        # Channel compression
        from_rgb_1 = tf.keras.layers.AveragePooling3D()(images)
        from_rgb_1 = tf.keras.layers.Conv3D(self._nf(self.current_resolution-1), kernel_size=1, padding='same', name='from_rgb_1')(from_rgb_1)

        from_rgb_2 = tf.keras.layers.Conv3D(self._nf(self.current_resolution), kernel_size=1, padding='same', name='from_rgb_2')(images)
        from_rgb_2 = e_block(from_rgb_2)

        lerp_input = self._weighted_sum()([from_rgb_1, from_rgb_2, alpha]) # RANDOM ALPHA

        # Getting latent code 
        e_z = self.growing_encoder(lerp_input)

        # Updating the model
        self.growing_encoder = tf.keras.Sequential([e_block,self.growing_encoder]) # without channel compression
        self.train_encoder = tf.keras.Model(inputs=[images,alpha],outputs=[e_z]) # with channel compression
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

    def make_Dblock(self,name):

        block_layers = []
        block_layers.append(tf.keras.layers.Flatten()) # obligé ??
        block_layers.append(tf.keras.layers.Dense(self.current_width**3)) 
        block_layers.append(tf.keras.layers.Reshape((self.current_width,self.current_width,self.current_width),input_shape=(self.current_width**3,)))
        #block_layers.append(tf.keras.layers.Activation(tf.nn.leaky_relu)) - depends on expression of NLL loss

        return tf.keras.models.Sequential(block_layers, name=name)

    def add_resolution(self):
        self.update_res()

        if self.current_resolution > 2:
            
            latent = tf.keras.layers.Input(shape=self.latent_size)
            alpha = tf.keras.layers.Input(shape=[], name='d_alpha')
            common = tf.keras.models.load_model(self.get_model(self.model_folder,self.current_resolution), custom_objects={'leaky_relu': tf.nn.leaky_relu}, compile=True)([latent,alpha])
            print(common.shape,tf.shape(common))
            shape = tf.shape(common)
            common = tf.reshape(common,[shape[0],common.shape[1],common.shape[2],common.shape[3]])
            print(common.shape)
            mu = self.make_Dblock(name='mu_block')(common)
            sigma = self.make_Dblock(name='sigma_block')(common)

            self.decoder = tf.keras.Model(inputs=[latent,alpha],outputs=[mu,sigma])
            self.decoder.trainable = True

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
        try:
            return str(list(path.glob('**/g_{}.h5'.format(res)))[0].resolve())
        except: print('No pretrained model for this resolution')

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def add_resolution(self):
        self.update_res()
        if self.current_resolution > 2:
            self.generator = tf.keras.models.load_model(self.get_model(self.model_folder,self.current_resolution), custom_objects={'leaky_relu': tf.nn.leaky_relu}, compile=True)
            self.generator.trainable = False

    def generate_latents(self,num_samples):
        latents = []
        for i in range(num_samples):
            latent = tf.random.normal((1, self.latent_size))
            latents.append(latent)
        return latents
import tensorflow as tf
from pathlib import Path
import numpy as np
import math
import dataset

class Encoder:

    def __init__(self, latent_size):
        super(Encoder, self).__init__()

        # static parameters 
        self.latent_size = latent_size
        self.num_channels = 1
        self.dimensionality = 3
        self.fmap_base =  2048 
        self.fmap_max = 8192

        # dynamic parameters
        self.current_resolution = 1
        self.current_width = 2 ** self.current_resolution
        self.growing_encoder = self.make_Ebase(nf=self._nf(1))
        self.train_encoder = tf.keras.Sequential(self.growing_encoder,name='sequential')
    
    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def update_weights(self):
        self.growing_encoder = tf.keras.Sequential()
        for layer in self.train_encoder.layers:
            if layer.name.startswith('block_') or layer.name.startswith('sequential'): 
                self.growing_encoder.add(layer)

    def make_Ebase(self,nf):

        # 2x2x2 images
        images = tf.keras.layers.Input(shape= (2,)*self.dimensionality + (nf,), name='images_iso2')

        # Final dense layer
        x = tf.keras.layers.Flatten()(images)
        x = tf.keras.layers.Dense(self.latent_size)(x)
        return tf.keras.models.Model(inputs=[images], outputs=[x], name='sequential')

    def make_Eblock(self,name,nf):
 
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
        self.update_weights()

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

        lerp_input = self._weighted_sum()([from_rgb_1, from_rgb_2, alpha])

        # Getting latent code 
        e_output = self.growing_encoder(lerp_input)

        # Updating the model
        self.growing_encoder = tf.keras.Sequential([e_block,self.growing_encoder]) # without channel compression
        self.train_encoder = tf.keras.Model(inputs=[images,alpha],outputs=[e_output]) # with channel compression
      
class Decoder(): 

    def __init__(self, latent_size, generator_folder):
        super(Decoder, self).__init__()
        
        # static parameters 
        self.latent_size = latent_size
        self.num_channels = 1
        self.dimensionality = 3
        self.fmap_base = 2048 
        self.fmap_max = 8192

        # dynamic parameters
        self.current_resolution = 1
        self.current_width = 2 ** self.current_resolution
        self.growing_decoder = self.make_Dbase(nf=self._nf(1))
        self.train_decoder = tf.keras.Sequential(self.growing_decoder,name='sequential')

    def get_model(self,folder,res):
        # find the model for the appropriate resolution
        path = Path(folder)
        return str(list(path.glob('g_{}.h5'.format(res)))[0].resolve()) # check if single one

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def _nf(self, stage): 
        # computes number of filters for each layer
        return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

    def _weighted_sum(self):
        return tf.keras.layers.Lambda(lambda inputs : (1-inputs[2])*inputs[0] + (inputs[2])*inputs[1])

    def make_Dbase(self,nf):

        latents = tf.keras.layers.Input(shape=[self.latent_size+self.label_size], name='latents')
        alpha = tf.keras.layers.Input(shape=[], dtype=tf.float32, name='g_alpha')

        # Latents stage
        x = tf.keras.layers.BatchNormalization(axis=-1)(latents)
        x = tf.keras.layers.Dense(self._nf(1)*(2**self.dimensionality))(x)
        x = tf.keras.layers.Reshape([2]*self.dimensionality+[self._nf(1)])(x)

        return tf.keras.models.Model(inputs=[latents, alpha], outputs=[x], name='decoder_base')

    def make_Dblock(self, nf, name=''):
        
        block_layers = []

        # block_layers.append(self.ConvTranspose(nf, kernel_size=3, strides=2, padding='same'))
        block_layers.append(tf.keras.layers.Upsampling3D())
        block_layers.append(tf.keras.layers.Convolution3D(nf, kernel_size=3, strides=1, padding='same'))
        block_layers.append(tf.keras.layers.Activation(tf.nn.leaky_relu))
        block_layers.append(tf.keras.layers.BatchNormalization(axis=-1))

        # block_layers.append(self.Conv(nf, kernel_size=3, strides=1, padding='same'))
        block_layers.append(tf.keras.layers.Convolution3D(nf, kernel_size=3, strides=1, padding='same'))
        block_layers.append(tf.keras.layers.Activation(tf.nn.leaky_relu))
        block_layers.append(tf.keras.layers.BatchNormalization(axis=-1))

        return tf.keras.models.Sequential(block_layers, name=name)

    def add_resolution(self):
        
        # Add resolution
        self.update_res()

        # Residual from input
        to_rgb_1 = tf.keras.layers.Upsampling3D()(self.growing_decoder.output)
        to_rgb_1 = tf.keras.layers.Convolution3D(self.num_channels, kernel_size=1)(to_rgb_1)
       
        # Growing generator
        d_block = self.make_Dblock(self._nf(self.current_resolution), name='d_block_{}'.format(self.current_resolution))
        d_block_output = d_block(self.growing_decoder.output)
        to_rgb_2 = tf.keras.layers.Convolution3D(self.num_channels, kernel_size=1)(d_block_output)

        lerp_output = self._weighted_sum()([to_rgb_1, to_rgb_2, self.growing_decoder.input[1]])
        d_output = tf.keras.layers.Activation('tanh')(lerp_output)

        # Updating the model
        self.growing_decoder = tf.keras.Sequential([self.growing_decoder,d_block]) # without channel compression
        self.train_decoder = tf.keras.Model(inputs=[self.growing_decoder.input],outputs=[d_output]) # with channel compression

    def get_decoder(self):
        return self.train_decoder

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
        


            


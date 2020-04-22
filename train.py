import tensorflow as tf
import tensorflow_probability as tfp
import losses
import networks
import dataset
import math
from pathlib import Path
import numpy as np


class PGVAE:

    def __init__(self,latent_size,generator_folder):

        self.strategy = tf.distribute.MirroredStrategy()

        # Dynamic parameters
        with self.strategy.scope():
            self.generator = networks.Generator(latent_size=latent_size,generator_folder=generator_folder)
            self.encoder = networks.Encoder(latent_size=latent_size)
            self.decoder = networks.Decoder(latent_size=latent_size,generator_folder=generator_folder)

        self.current_resolution = 1
        self.current_width = 2**self.current_resolution
        self.res_batch = {2:128,4:64,8:32,16:16,32:8,64:4,128:2,256:1}
        self.res_epoch = {2:10,4:20,8:40,16:60,32:80,64:100,128:200,256:400}

        # Static parameters
        self.generate = True
        self.prior = 'Normal'
        self.learning_rate = 0.001

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def add_resolution(self):
        with self.strategy.scope():
            self.update_res()
            self.generator.add_resolution()
            self.encoder.add_resolution()
            self.decoder.add_resolution() 

    def get_current_alpha(self, iters_done, iters_per_transition):
        return iters_done/iters_per_transition

    def get_batchsize(self):
        return self.res_batch[self.current_width]

    def get_epochs(self):
        return self.res_epoch[self.current_width]

    def train_resolution(self,batch_size,epochs,save_folder):

        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync),flush=True)
        global_batch_size = batch_size * self.strategy.num_replicas_in_sync

        # Chack points 
        savefolder = Path(save_folder)
        checkpoint_prefix = savefolder.joinpath("vae{}.ckpt".format(self.current_resolution))

        # create dataset 
        data = self.generator.generate_latents(num_samples=100)
        ds = dataset.get_dataset(data,batch_size)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(ds)

        # Training loops
        with self.strategy.scope():

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8) # QUESTIONS PARAMETERS
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.encoder.train_encoder)

            def train_step(inputs,alpha):
                with tf.GradientTape() as tape:

                    # Forward pass 
                    images = self.generator.generator([inputs,alpha],training=False)
                    latent_codes = self.encoder.train_encoder(images,training=True)
                    reconst_images = self.decoder.decoder([latent_codes,alpha],training=False)
                    
                    # Forward pass - Variational
                    #mu, sigma = self.encoder.train_encoder(inputs,training=True)
                    #latent_code = tfp.distributions.Normal(loc=mu, scale=sigma) #/ tfd.MultivariateNormalDiag(loc, scale)/tfd = tf.contrib.distributions
                    #reconst_images = self.decoder.decoder(latent_code,training=False)

                    # Compute the ELBO loss for VAE training 
                    #reconstruction = losses.Reconstruction_loss(true=inputs,predict=reconst_images)
                    #kl = losses.Kullback_Leibler(mu=mu,sigma=sigma)
                    #elbo = losses.ELBO(kl=kl,reconstruction=reconstruction)
                    #elbo = tf.nn.compute_average_loss(elbo, global_batch_size=global_batch_size)

                    # Compute the reconstruction loss for AE training
                    error = losses.Reconstruction_loss(true=images,predict=reconst_images)

                # Backward pass for AE
                #grads = tape.gradient(elbo,self.encoder.train_encoder.trainable_variables) - VAE
                grads = tape.gradient(error, self.encoder.train_encoder.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.encoder.train_encoder.trainable_variables))
                
                # return elbo
                return error

            @tf.function
            def distributed_train_step(inputs,alpha):
                per_replica_losses = self.strategy.experimental_run_v2(train_step, args=(inputs,alpha,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) # axis

        # Start training.
        for epoch in range(self.res_epoch[self.current_width]):
            print('Starting the training : epoch {}'.format(epoch),flush=True)
            total_loss = 0.0
            num_batches = 0

            alpha = tf.constant(self.get_current_alpha(epoch,self.res_epoch[self.current_width]),tf.float32)
            print('Alpha: ',alpha)

            for this_latent in train_dist_dataset:
                tmp_loss = distributed_train_step(this_latent,alpha)
                total_loss += tmp_loss
                num_batches += 1
                print('----- Batch Number {} : {}'.format(num_batches,tmp_loss),flush=True)

            train_loss=total_loss/num_batches

            # save results
            checkpoint.save(checkpoint_prefix)
            template = ("Epoch {}, Loss: {}")
            print (template.format(epoch+1, train_loss),flush=True)

        #Save the model and the history
        self.encoder.train_encoder.save(savefolder.joinpath('e{}.h5'.format(self.current_resolution)))

    def train(self,stop_res,save_folder,start_res=2):

        start_stage = math.log(2,2)
        stop_stage = math.log(stop_res,2) # check if multiple of 2

        resolutions = [2**x for x in np.arange(start_stage+1,stop_stage+1)]

        for i, resolution in enumerate(resolutions):
            print('Processing step {}: resolution {} with max resolution {}'.format(i,resolution,resolutions[-1]),flush=True)
            
            self.add_resolution()

            batch_size = self.get_batchsize()
            epochs = self.get_epochs()

            if self.current_resolution > 2: self.train_resolution(batch_size,epochs,save_folder)




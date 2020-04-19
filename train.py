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

        self.generator = networks.Generator(latent_size=latent_size,generator_folder=generator_folder)
        self.encoder = networks.Encoder(latent_size=latent_size)
        self.decoder = networks.Decoder(latent_size=latent_size,generator_folder=generator_folder)

        self.current_resolution = 1
        self.current_width = 2**self.current_resolution
        self.res_batch = [128,64,32,16,8,4,2,1]

        self.generate = True
        self.strategy = True
        self.prior = 'Normal'
        self.learning_rate = 0.001

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2 ** self.current_resolution

    def add_resolution(self):
        self.update_res()
        self.generator.add_resolution()
        self.encoder.add_resolution()
        self.decoder.add_resolution() 

    def get_batchsize(self):
        return self.res_batch[self.current_resolution-1]

    def train_resolution(self,tf_folder,batch_size,epochs,save_folder):

        if self.generate : self.generator.generate_samples(10,tf_folder) # for the moment dataset = images, TO DO datasets = latents

        strategy = tf.distribute.MirroredStrategy()
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync),flush=True)
        global_batch_size = batch_size * strategy.num_replicas_in_sync

        # Chack points 
        savefolder = Path(save_folder)
        checkpoint_prefix = savefolder.joinpath("vae{}.ckpt".format(self.current_resolution))

        # paralellize the dataset 
        tfdataset = dataset.get_dataset(tf_folder,batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(tfdataset)

        # Training loops
        with strategy.scope():

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8) # QUESTIONS PARAMETERS
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.encoder)

            def train_step(inputs):
                with tf.GradientTape() as tape:
                    
                    # Forward pass
                    mu,sigma = self.encoder.train_encoder(inputs,training=True)
                    latent_code = tfp.distributions.Normal(loc=mu, scale=sigma)
                    reconst_images = self.decoder.decoder(latent_code,training=False)

                    # Compute the ELBO loss for VAE training 
                    reconstruction = losses.Reconstruction_loss(true=inputs,predict=reconst_images)
                    kl = losses.Kullback_Leibler(mu=mu,sigma=sigma)
                    elbo = losses.ELBO(kl=kl,reconstruction=reconstruction)
                    elbo = tf.nn.compute_average_loss(elbo, global_batch_size=global_batch_size)

                # Backward pass
                grads = tape.gradient(elbo, self.encoder.train_encoder.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.encoder.train_encoder.trainable_variables))
                return elbo

            @tf.function
            def distributed_train_step(inputs):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) # axis

        # Start training.
        for epoch in range(epochs):
            print('Starting the training : epoch {}'.format(epoch),flush=True)
            total_loss = 0.0
            num_batches = 0
            for this_x in train_dist_dataset:
                tmp_loss = distributed_train_step(this_x)
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

        return 1

    def train(self,tf_folder,stop_res,epochs,save_folder,start_res=2):

        start_stage = math.log(2,2)
        stop_stage = math.log(stop_res,2) # check if multiple of 2

        resolutions = [2**x for x in np.arange(start_stage+1,stop_stage+1)]

        for i, resolution in enumerate(resolutions):
            print('Processing step {}: resolution {} with max resolution {}'.format(i,resolution,stop_res),flush=True)
            batch_size = self.get_batchsize()

            self.add_resolution()
            self.train_resolution(tf_folder,batch_size,epochs,save_folder)




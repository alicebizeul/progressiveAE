import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import losses
import Vnetworks
import dataset
import utils
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class PGVAE:

    def __init__(self,latent_size,generator_folder,restore,param_optimizer):

        self.strategy = tf.distribute.MirroredStrategy()

        # Dynamic parameters
        with self.strategy.scope():
            self.generator = Vnetworks.Generator(latent_size=latent_size,generator_folder=generator_folder)
            self.encoder = Vnetworks.Encoder(latent_size=latent_size)
            self.decoder = Vnetworks.Decoder(latent_size=latent_size,generator_folder=generator_folder)

        self.current_resolution = 1
        self.current_width = 2**self.current_resolution
        self.res_batch = {2:64,4:32,8:16,16:8,32:4,64:2,128:1,256:1}
        self.res_epoch = {2:10,4:10,8:30,16:0,32:80,64:100,128:200,256:400}

        # Static parameters
        self.generate = True
        self.learning_rate = 0.0001
        self.latent_size = 1024
        self.restore = False
        self.optimizer = param_optimizer

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

    def reparametrization_trick(self,mu,sigma):
        epsilon = tfd.Independent(tfd.Normal(loc=tf.zeros(self.latent_size), scale=tf.ones(self.latent_size)))
        print(sigma.shape)
        return mu + tf.exp(sigma) * epsilon.sample(1)

    def train_resolution(self,dataset,batch_size,epochs,save_folder,num_samples):

        # Check points 
        savefolder = Path(save_folder)
        checkpoint_prefixE = savefolder.joinpath("vae_e{}.ckpt".format(self.current_resolution))
        checkpoint_prefixD = savefolder.joinpath("vae_d{}.ckpt".format(self.current_resolution))

        # Training loops
        with self.strategy.scope():

            # Initialise
            if self.optimizer=='Adam':optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
            if self.optimizer=='AdaMod':optimizer = utils.AdaMod(learning_rate=0.001,beta_1=0.9,beta_2=0.999,beta_3=0.9995,epsilon=1e-8)
            
            checkpointE = tf.train.Checkpoint(optimizer=optimizer, model=self.encoder.train_encoder)
            checkpointD = tf.train.Checkpoint(optimizer=optimizer, model=self.decoder.decoder)

            if self.restore and self.current_resolution == 4: 
                #print(self.encoder.train_encoder.get_weights())
                #latest = tf.train.latest_checkpoint(save_folder)
                checkpointE.restore(save_folder+'vae4.ckpt-60')
                #print(self.encoder.train_encoder.get_weights())

            def train_step(inputs,alpha):
                with tf.GradientTape() as tape:

                    # Forward pass 
                    images = self.generator.generator([inputs,alpha],training=False)
                    q = self.encoder.train_encoder([images,alpha],training=True)
                    #print('q mu',q[:,:1024])
                    #print('q sigma',q[:,1024:])
                    z = self.reparametrization_trick(mu=q[:,:1024],sigma=q[:,1024:])
                    #print('z',z)
                    [p_mu, p_log_sigma] = self.decoder.decoder([z,alpha],training=True)
                    #print('p mu',p_mu)
                    #print('p sigma',p_log_sigma)

                    # ELBO Error computation 
                    nll = losses.neg_loglikelihood(true=images,predict_mu=p_mu,predict_log_sigma=p_log_sigma,var_epsilon=1)
                    print('nll',nll)
                    kl = losses.Kullback_Leibler(mu=q[:,:1024],log_sigma=q[:,1024:])
                    print('kl',kl)
                    error = losses.ELBO(neg_log_likelihood=nll,kl=kl)
                    global_error = tf.nn.compute_average_loss(error, global_batch_size=batch_size) # recheck

                grads = tape.gradient(global_error, self.encoder.train_encoder.trainable_variables+self.decoder.decoder.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.encoder.train_encoder.trainable_variables+self.decoder.decoder.trainable_variables))
                
                return global_error

            #@tf.function
            def distributed_train_step(inputs,alpha):
                per_replica_losses = self.strategy.experimental_run_v2(train_step, args=(inputs,alpha,))
                
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) # axis

        # Start training.
        print('EPOCH ',self.res_epoch[self.current_width])
        for epoch in range(self.res_epoch[self.current_width]):
            print('Starting the training : epoch {}'.format(epoch),flush=True)
            total_loss = 0.0
            num_batches = 0

            alpha = tf.constant(self.get_current_alpha(epoch,self.res_epoch[self.current_width]),tf.float32) # increases with the epochs

            for this_latent in dataset:
                tmp_loss = distributed_train_step(this_latent,alpha)
                total_loss += tmp_loss
                num_batches += 1
                if num_batches%50 == 0: 
                    print('----- Batch Number {} : {}'.format(num_batches,tmp_loss),flush=True)

            train_loss=total_loss/num_batches

            # save results
            checkpointE.save(checkpoint_prefixE)
            checkpointD.save(checkpoint_prefixD)
            template = ("Epoch {}, Loss: {}")
            print (template.format(epoch+1, train_loss),flush=True)

        #Save the model and the history
        self.encoder.train_encoder.save(savefolder.joinpath('e{}.h5'.format(self.current_resolution)))
        self.decoder.decoder.save(savefolder.joinpath('d{}.h5'.format(self.current_resolution)))

    def train(self,stop_width,save_folder,tf_folder,start_width,num_samples):

        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync),flush=True) 

        start_res = math.log(start_width,2)
        stop_res = math.log(stop_width,2) # check if multiple of 2

        resolutions = [2**x for x in np.arange(2,stop_res+1)]

        for i, resolution in enumerate(resolutions):
            print('Processing step {}: resolution {} with max resolution {}'.format(i,resolution,resolutions[-1]),flush=True)
            
            self.add_resolution()

            batch_size = self.get_batchsize()
            global_batch_size = batch_size * self.strategy.num_replicas_in_sync
            epochs = self.get_epochs()

            batched_dataset = self.generator.generate_latents(num_samples=num_samples)
            batched_dist_dataset = self.strategy.experimental_distribute_dataset(dataset.get_dataset(batched_dataset,global_batch_size))

            print('**** Batch size : {}   | **** Epochs : {}'.format(batch_size,epochs))

            if self.current_resolution >= start_res and self.current_resolution > 2: 
                self.train_resolution(batched_dist_dataset,global_batch_size,epochs,save_folder,num_samples)

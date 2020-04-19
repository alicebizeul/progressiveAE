import tensorflow as tf
import tensorflow_probability as tfp

def Kullback_Leibler(mu,sigma):

    q = tfp.distributions.Normal(loc=mu,scale=sigma,name='Estimate')
    p = tfp.distributions.Normal(loc=0,scale=1,name='True')

    # measure the kl divergence over all dimensions and then sum the values
    return tf.reduce_sum(tfp.distributions.kl_divergence(p,q,name='KL divergence'),1)

def Reconstruction_loss(true,predict):
    return tf.reduce_mean(tf.square(tf.subtract(true, predict))) # sum over all dimensions

def ELBO(kl,reconstruction):
    return tf.add(kl,reconstruction)
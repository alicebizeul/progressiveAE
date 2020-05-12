import tensorflow as tf
import tensorflow_probability as tfp
import math
#from keras import backend as K

def Kullback_Leibler(mu,sigma):

    q = tfp.distributions.Normal(loc=mu,scale=sigma,name='Estimate')
    p = tfp.distributions.Normal(loc=0,scale=1,name='True')

    # measure the kl divergence over all dimensions and then sum the values
    return tf.reduce_sum(tfp.distributions.kl_divergence(p,q,name='KL divergence'),1)

def neg_loglikelihood(true,predict_mu,predict_sigma,var_epsilon):

    # Gaussian prior, independant pixels
    #loss = ( 0.5 * math.log(2 * math.pi)
    #        + 0.5 * K.log(predict_sigma + var_epsilon)
    #        + 0.5 * K.square(true - predict_mu) / (predict_sigma + var_epsilon))
    #print(loss.shape)
    #return loss
    return 1
    
def Reconstruction_loss(true,predict):
    return tf.reduce_mean(tf.square(tf.subtract(true, predict)),axis=[1,2,3]) # sum over all dimensions

def ELBO(kl,neg_log_likelihood):
    return tf.add(kl,neg_log_likelihood)
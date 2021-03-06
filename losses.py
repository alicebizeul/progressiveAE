import tensorflow as tf
import tensorflow_probability as tfp
import math
#from keras import backend as K

def Kullback_Leibler(mu,log_sigma):

    # or sum over divergence for each normal distribution
    q = tfp.distributions.MultivariateNormalDiag(loc=mu,scale_diag=tf.exp(log_sigma),name='Estimate')
    p = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(1024),scale_diag=tf.ones(1024),name='True')

    # measure the kl divergence over all dimensions and then sum the values
    return tfp.distributions.kl_divergence(p,q,name='KLdivergence')
    #return tfp.distributions.kl_divergence(p,q,name='KL divergence')
    
def neg_loglikelihood(true,predict_mu,predict_log_sigma,var_epsilon):

    # Gaussian prior, independant pixels
    print('detailed nll' ,predict_log_sigma,tf.math.log(predict_log_sigma + var_epsilon))
    loss = ( 0.5 * tf.math.log(predict_log_sigma + var_epsilon)+ 0.5 * tf.square(tf.math.subtract(tf.squeeze(true) , predict_mu)) / (predict_log_sigma + var_epsilon))
    return tf.math.reduce_sum(loss,axis=[1,2,3])
    
def Reconstruction_loss(true,predict):
    return tf.reduce_mean(tf.square(tf.subtract(true, predict)),axis=[1,2,3]) # sum over all dimensions

def ELBO(kl,neg_log_likelihood):
    return tf.add(kl,neg_log_likelihood)
import os
#import Vtrain
import train
from options import Opts

def main(config):

    if config.task == 'train':
        if not config.variational: 
            pgvae = train.PGVAE(latent_size=config.latent,generator_folder=config.generator_folder,restore=config.restore,param_optimizer=config.param_optimizer) # make sure strategy is true when multiple GPUs available
            pgvae.train(start_width=config.start_res, stop_width=config.stop_res,save_folder=config.save_folder,tf_folder=config.tf_folder,num_samples=config.num_samples)
        #if config.variational: 
        #    pgvae = Vtrain.PGVAE(latent_size=config.latent,generator_folder=config.generator_folder,restore=config.restore,param_optimizer=config.param_optimizer) # make sure strategy is true when multiple GPUs available
        #    pgvae.train(start_width=config.start_res, stop_width=config.stop_res,save_folder=config.save_folder,tf_folder=config.tf_folder,num_samples=config.num_samples)

if __name__ == '__main__':
    opt = Opts()
    config = opt.parse()

    main(config)
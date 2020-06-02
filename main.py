import os
#from Vtrain import PGVAE
from train import PGVAE
from options import Opts

def main(config):

    if config.task == 'train':
        pgvae = PGVAE(latent_size=config.latent,generator_folder=config.generator_folder,restore=config.restore,param_optimizer=config.param_optimizer) # make sure strategy is true when multiple GPUs available
        pgvae.train(start_width=config.start_res, stop_width=config.stop_res,save_folder=config.save_folder,num_samples=config.num_samples)

if __name__ == '__main__':
    opt = Opts()
    config = opt.parse()

    main(config)
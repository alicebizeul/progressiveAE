import os
from train import PGVAE
from options import Opts

def main(config):

    if config.task == 'train':
        pgvae = PGVAE(latent_size=config.latent,generator_folder=config.generator_folder)
        pgvae.train(tf_folder=config.tf_folder,start_res = config.start_res, stop_res = config.stop_res,epochs=config.epochs,save_folder=config.save_folder)

if __name__ == '__main__':
    opt = Opts()
    config = opt.parse()

    main(config)
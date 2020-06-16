import argparse
import os

class Opts:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='From a GAN to an Variational Autoencoder')
        self.subparsers = self.parser.add_subparsers(dest='task', help='train')
        
        # Train the model progressivly
        self.train = self.subparsers.add_parser('train',help='Train VAE progressivly')
        self.train.add_argument('--generator_folder',required=True,type=str)
        self.train.add_argument('--latent', required=False, type=int, default=1024)
        self.train.add_argument('--start_res',required=False,type=int,default=2)
        self.train.add_argument('--stop_res',required=True,type=int)
        self.train.add_argument('--save_folder',required=True,type=str)
        self.train.add_argument('--tf_folder',required=True,type=str)
        self.train.add_argument('--num_samples',required=False,type=int,default=50000)
        self.train.add_argument('--restore',required=False,type=bool,default=False)
        self.train.add_argument('--param_optimizer',required=False,type=str,default="Adam")
        self.train.add_argument('--variational',required=False,type=bool,default=False)

    def parse(self):
        config = self.parser.parse_args()

        return config

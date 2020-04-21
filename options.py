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

    def parse(self):
        config = self.parser.parse_args()

        return config

import argparse
import os

class Opts:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='From a GAN to an Variational Autoencoder')
        self.subparsers = self.parser.add_subparsers(dest='task', help='train')
        
        # Train the model progressivly
        self.train = self.subparsers.add_subparsers(dest='Train VAE progressivly')
        self.train.add_subparsers('-- generator_folder',required=True,typr=str)
        self.train.add_subparsers('--latent', required=False, type=int, default=10)
        self.train.add_subparsers('--start_res',required=False,type=int,default=2)
        self.train.add_subparsers('--stop_res',required=True,type=int)
        self.train.add_subparsers('--epochs',required=False,type=int,default=10)
        self.train.add_subparsers('--save_folder',required=True,type=str)
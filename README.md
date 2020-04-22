# Progressive Autoencoder

Progressive training of Variational Autoencoder for the generation of 3D MRI brains.

 ## Installation

 For the installation of the environment : 

      $ pip install -r requirements.txt 

## Training 

To progressivly train the AE using pretrained generators :

      $ python3 main.py train --generator_folder mygenerators/ --save_folder mymodels/ --stop_res 4 
Required arguments are:

- **generator_folder** : folder to the _.h5_ pre-trained models to be used as the image generator & AE decoder
- **save_folder** : folder where trained AE should be stored
- **stop_res** : final image resolution to achieve during training (e.g: 2, 4, 8, 16, 32, 64, 128, 256)

Additional arguments : 

- **start_res** : initial image resolution to start training with (default: 2)
- **latent** : latent code size (default: 1024)

## Testing 

Not Implemented Yet

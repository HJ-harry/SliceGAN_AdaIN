'''
General structure of the code adopted from: https://github.com/stke9/SliceGAN
with slight modifications to dataloading process, and the generator.
You may change the training configurations in this script.
'''

from slicegan import model_disentangle, networks, util
import argparse
# Define project name
Project_name = 'disentangle'
# Project_name = 'disentangle_trainnew'
# Specify project folder.
Project_dir = 'checkpoints/battery_cathode'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
parser.add_argument('training', type=int)
args = parser.parse_args()
Training = args.training
Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'threephase'
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif', 'png', 'jpg','array')
data_type = 'tif'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['data/battery_cathode/synth_data/synth_015/synth_015.mat',
             'data/battery_cathode/synth_data/synth_035/synth_035.mat',
             'data/battery_cathode/synth_data/synth_060/synth_060.mat']

if Training:
    codes = [15.0, 35.0, 60.0]
else:
    # In the test phase, you may choose any scalar value within the range that you have used to train the network
    codes = [15.0, 17.5, 20.0, 22.5, 25.0, 30.0, 35.0, 60.0]

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, img_channels, scale_factor = 64, 3, 1
# z vector depth
z_channels = 16
# Layers in G and D
lays = 6
# kernals for each layer
dk, gk = [4]*lays, [4]*lays
# strides
ds, gs = [2]*lays, [2]*lays
# no. filters
df, gf = [img_channels, 64, 128, 256, 512, 1], [z_channels,512, 256, 128, 64, img_channels]
# paddings
dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

## Create Networks
netD, netG = networks.slicegan_nets_disentangle(Project_path, Training, image_type, dk, ds, df,dp, gk ,gs, gf, gp)

# Train
if Training:
    model_disentangle.train(Project_path, image_type, data_type, data_path, codes, netD, netG, img_channels, img_size, z_channels, scale_factor)
else:
    img, raw, netG = util.test_img_disentangle(Project_path, image_type, netG(), codes, z_channels, lf=6, periodic=[0, 1, 1])
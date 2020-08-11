from skimage.util import view_as_windows
from matplotlib import image
from PIL import Image
import numpy as np
import argparse
import os
import math

def patchify(patches, patch_size, step):
    return view_as_windows(patches, patch_size, step)

#ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("Origin_Directory", help="Absolute path of directory where the data is",
                    type=str)
parser.add_argument("Destination_Directory", help="Absolute path of directory where you want your patchified data to be created",
                    type=str)
parser.add_argument("Number_tiles", help="Determines the number of tiles each image is divided in",
                    type=int)
args = parser.parse_args()

num_tiles = args.Number_tiles
ORGN_PATH = str(args.Origin_Directory)
DESTIN_PATH = str(args.Destination_Directory)

#Define args
sizes = ['/4x20x', '/20x100x']
types = ['/train','/test']
resolutions = ['/low_res','/high_res', '/high_res2']

#Fix window size & step size
IMG_HEIGHT, IMG_WIDTH, IMG_CHNNL = (2160, 2560, 3)

divider = math.sqrt(num_tiles)
window_shape = (int(IMG_HEIGHT//divider), int(IMG_WIDTH//divider), 3)

step_size = (int(window_shape[0]//2), int(window_shape[1]//2), 3) #(108,128)
#step_size = (180, 160, 3) #(180,160)

#Patchify

def patchify_images(size, type, res, num_tiles):

    #Define original path and goes there
    ORG_PATH = ORGN_PATH + size + type + res
    os.chdir(ORG_PATH)

    #List images in this path & pre-process list
    list_images = os.listdir()
    if ".DS_Store" in list_images:
        list_images.remove('.DS_Store')

    #Checks extension 'tiff' vs 'tif'
    extension = '.' + list_images[0].split('.')[1]

    for i in range(len(list_images)):
        list_images[i] = list_images[i].split('.')[0]

    #Destination path
    DEST_PATH = DESTIN_PATH + size + type + res

    #Check destination and create path if necessary
    if not os.path.exists(DEST_PATH):
        os.makedirs(DEST_PATH)

    #Goes back to origin path
    #os.chdir(ORG_PATH)

    #Split and save
    for i in range(len(list_images)):
        PATH = list_images[i] + extension
        img = image.imread(PATH)
        patches = patchify(img, window_shape, step=step_size)

        len_height ,len_width = patches.shape[0], patches.shape[1]

        for k in range(len_height):
            for l in range(len_width):
                patch = np.squeeze(patches[k][l])
                im = Image.fromarray(patch)
                im.save(f"{DEST_PATH}/{list_images[i]}_{k:02d}_{l:02d}.tiff")

for size in sizes:
    for type in types:
        for res in resolutions:
            try:
                patchify_images(size, type, res, num_tiles)
            except Exception:
                pass

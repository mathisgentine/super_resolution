import image_slicer
import argparse
import os

#ArgumentParser
parser = argparse.ArgumentParser()

parser.add_argument("Origin_Directory", help="Absolute path of directory where the data is",
                    type=str)
parser.add_argument("Destination_Directory", help="Absolute path of directory where you want your sliced data to be created",
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

def slice_images(size, type, res, num_tiles):

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
        img = list_images[i] + extension
        tiles = image_slicer.slice(img, num_tiles, save=False)
        image_slicer.save_tiles(tiles, directory=DEST_PATH, format='tiff', prefix=list_images[i])

for size in sizes:
    for type in types:
        for res in resolutions:
            try:
                slice_images(size, type, res, num_tiles)
            except Exception:
                pass

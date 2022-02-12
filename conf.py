import numpy as np
import sys
from PIL import Image
import random
import pandas as pd
import math

def get_image_info(image_name):
    """
    Returns the path of the input image and some more information for training.
    As written in the paper, the method uses repeating object window for rescaling the image 
    and initialize the positive bucket.
    This was done by asking the user to mark a window, however, in this version it is hard-coded here.
    :param image_name: name of the image from the dataset (as written in the table in the paper)
    """

    th = 0.85
    number_of_patches = 8
    show_gray = 0
    max_num_cells = 400
    gt_color = [255,255,255]
    dir = "/content/drive/My Drive/Colab Notebooks/INBARHUB/dataset/"
    patch_sz = 21
    if "Cell" in image_name:
        th = 0.85
        number_of_patches = 8
        show_gray = 0
        max_num_cells = 400
        gt_color = [255,255,255]
        dir = "/content/drive/My Drive/Colab Notebooks/INBARHUB/nucls_data/"
        patch_sz = 25
        path = dir+image_name+".png"
        img = Image.open(path)
        width = int(math.floor(img.width/10))*10
        height = int(math.floor(img.height/10))*10
        im_size = [height,width]

    else:
        print(image_name + ' not found image name')
        return None
    
    return [im_size,number_of_patches,th,path,show_gray,max_num_cells,gt_color,patch_sz]


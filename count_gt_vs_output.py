from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
from conf import get_image_info
import pandas as pd

import numpy as np
import scipy.misc
import os
import os.path

def get_gt_data(name):
    [im_size,_,_,gt_image_name,_,_,gt_color,patch_sz] = get_image_info(name)
    gt_image = np.array(Image.open(gt_image_name.split('.')[0]+"_gt."+gt_image_name.split('.')[1]))
    gt_image = gt_image[:,:,:3]
    print(gt_image_name.split('.')[0]+"_gt."+gt_image_name.split('.')[1])

    substring = "Cell"
    if substring in name: 
        row_gt, col_gt = get_gt_data_csv(gt_image_name)
    else:
        row_gt, col_gt = get_gt_data_image(gt_image,gt_color)

    ratio = [im_size[0]/np.float(gt_image.shape[0]),im_size[1]/np.float(gt_image.shape[1])]
    patch_sz_hf = patch_sz//2
    row_gt = np.int32(row_gt*ratio[0]) + patch_sz_hf
    col_gt = np.int32(col_gt*ratio[1]) + patch_sz_hf
    print(f'loading {len(row_gt)} ground truth points from {gt_image_name}')
    return row_gt, col_gt, ratio

def get_gt_data_image(gt_image,gt_color):
    # Read the locations from the ground-truth image
    tmp = np.where(np.reshape(gt_image[:,:,0],-1) == gt_color[0])[0]
    tot_tmp = tmp
    tmp = np.where(np.reshape(gt_image[:,:,1],-1) == gt_color[1])[0]
    tot_tmp = np.intersect1d(tmp, tot_tmp)
    tmp = np.where(np.reshape(gt_image[:,:,2],-1) == gt_color[2])[0]
    tot_tmp = np.intersect1d(tmp, tot_tmp)
    [row_gt,col_gt] = np.unravel_index(tot_tmp, [gt_image.shape[0], gt_image.shape[1]])
    return row_gt, col_gt

def get_gt_data_csv(gt_image_name): 
    gt_data = pd.read_csv(gt_image_name.split('.')[0]+"_gtdata.csv")
    group_data = gt_data.loc[gt_data["group"] == "lymphocyte"]

    x_min = group_data[['xmin']].to_numpy()
    x_max = group_data[['xmax']].to_numpy()
    col_gt = np.int32((x_min + x_max)/2).flatten()

    y_min = group_data[['ymin']].to_numpy()
    y_max = group_data[['ymax']].to_numpy()
    row_gt = np.int32((y_min + y_max)/2).flatten()
    return row_gt, col_gt

def count_images(row_gt,col_gt,dir,step,patch_sz,im_size):
    """
    This function evaluates the current solution - false positive and negative, and the total count.
    The method loads the current soltuion as well as the ground-truth image and calaulates the performance
    :param name: name of the image from the dataset (as written in the table in the paper)
    :param dir: the directory where the output is saved (it is declared before calling to main function)
    :param step: number of iteration
    :param patch_sz: the size of the repeating object
    """

    '''
    [im_size,_,_,_,gt_image_name,_,_,gt_color,_] = get_image_info(name)
    gt_image = np.array(Image.open(gt_image_name.split('.')[0]+"_gt."+gt_image_name.split('.')[1]))
    gt_image = gt_image[:,:,:3]
    print(gt_image_name.split('.')[0]+"_gt."+gt_image_name.split('.')[1])

    # Read the locations of the repeating object from our solution
    res = np.loadtxt(dir + 'res_ours_' + str(step) + '.txt')
    [row_ours,col_ours] = np.unravel_index(np.int32(res), [im_size[0], im_size[1]])

    # Read the locations from the ground-truth image
    tmp = np.where(np.reshape(gt_image[:,:,0],-1) == gt_color[0])[0]
    tot_tmp = tmp
    tmp = np.where(np.reshape(gt_image[:,:,1],-1) == gt_color[1])[0]
    tot_tmp = np.intersect1d(tmp, tot_tmp)
    tmp = np.where(np.reshape(gt_image[:,:,2],-1) == gt_color[2])[0]
    tot_tmp = np.intersect1d(tmp, tot_tmp)
    [row_gt,col_gt] = np.unravel_index(tot_tmp, [gt_image.shape[0], gt_image.shape[1]])

    ratio = [np.float(gt_image.shape[0])/im_size[0],np.float(gt_image.shape[1])/im_size[1]]
    row_gt = np.int32(row_gt/ratio[0])
    col_gt = np.int32(col_gt/ratio[1])
    '''

    #row_gt, col_gt = get_gt_data(name)

    # Read the locations of the repeating object from our solution
    res = np.loadtxt(dir + 'res_ours_' + str(step) + '.txt')
    [row_ours,col_ours] = np.unravel_index(np.int32(res), [im_size[0], im_size[1]])

    sz = 3

    matched_pt = []
    gt_count = row_gt.shape
    count = row_ours.shape
    row_ours_copy = row_ours.copy()
    col_ours_copy = col_ours.copy()
    row_gt_copy = row_gt.copy()
    col_gt_copy = col_gt.copy()

    # The repeating object locations of the ground truth and our locations should be close
    # by 2\3*path_sz in order to be counted
    r = 2*patch_sz/3
    counter = 0
    for i in range(0,len(row_gt)):
        match = []
        for j in range(0,len(row_ours)):
            if np.power(row_gt[i]-row_ours[j],2)+np.power(col_gt[i] - col_ours[j],2) < np.power(r,2) and row_ours_copy[j] != -1:
                match.append([j,np.power(row_gt[i]-row_ours[j],2)+np.power(col_gt[i]-col_ours[j],2)])
        if len(match) > 0:
            counter = counter+1
            match = np.stack(match)
            closer = np.argmin(match[:,1])
            row_ours_copy[match[closer,0]] = -1
            matched_pt.append(i)
    if len(matched_pt)>0:
        matched_pt = np.stack(matched_pt)
    row_gt_copy = np.delete(row_gt_copy,matched_pt)
    col_gt_copy = np.delete(col_gt_copy,matched_pt)
    col_ours_copy = np.delete(col_ours_copy,np.where(row_ours_copy == -1)[0])
    row_ours_copy = np.delete(row_ours_copy,np.where(row_ours_copy == -1)[0])

    return [row_ours_copy.shape,row_gt_copy.shape,gt_count,count]

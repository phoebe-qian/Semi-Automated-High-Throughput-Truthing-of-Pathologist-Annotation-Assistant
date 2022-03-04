from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
from conf import get_image_info
import pandas as pd

import numpy as np
import scipy.misc
import os
import os.path

def normalize_image(x):
    xmin = np.min(x.flatten())
    xmax = np.max(x.flatten())
    return (x - xmin)/(xmax-xmin)

def purple_dots_on_image(input,all_center_ind,ind_to_ask,acc_ind_final,patch_sz_hf):
    """
    Create the method's solution image. This method outputs the input image with the locations of the repeating objects
    """
    final_res_image = np.zeros((np.shape(input)[1],np.shape(input)[2],3))
    gt_color_1 = [0.64,0.27,0.61]

    for i in range(0,3):
        final_res_image[:,:,i] = (input[0,:,:,0]+input[0,:,:,1]+input[0,:,:,2])/3

    for i in range(0,len(ind_to_ask)):
        r=all_center_ind[ind_to_ask[i]][0]
        c=all_center_ind[ind_to_ask[i]][1]
        final_res_image[r-2:r+2,c-2:c+2] = gt_color_1

    acc_final = np.where(acc_ind_final == 1)[0]
    for i in range(0,len(acc_final)):        
        r=all_center_ind[acc_final[i]][0]
        c=all_center_ind[acc_final[i]][1]
        final_res_image[r-2:r+2,c-2:c+2] = gt_color_1

    final_res_image = final_res_image[patch_sz_hf:-patch_sz_hf,patch_sz_hf:-patch_sz_hf,:]
    count = len(acc_final)+len(np.setdiff1d(ind_to_ask,acc_final))

    return [count,final_res_image]

def show_point_on_image(input,row_col,color = [0.64,0.27,0.61],convert_gray=True):
    """
    Create image with points in col_row marked by pt_color. This method outputs the input image with the locations of the repeating objects
    """
    pt_image = np.zeros((np.shape(input)[0],np.shape(input)[1],3))
    if convert_gray:
        for i in range(0,3):
            pt_image[:,:,i] = (input[:,:,0]+input[:,:,1]+input[:,:,2])/3
    else:
        for i in range(0,3):
            pt_image[:,:,i] = input[:,:,i]

    for p in row_col:
        r,c=p
        pt_image[r-2:r+2,c-2:c+2] = color
   
    #pt_image = pt_image[patch_sz_hf:-patch_sz_hf,patch_sz_hf:-patch_sz_hf,:]

    return pt_image

def show_acc_rej_on_image(input_im,all_center_ind,acc_ind_final,rej_ind_final,out_image_file,convert_gray=True):
    acc_pt = all_center_ind[np.where(acc_ind_final==1)] #(row,col)
    rej_pt = all_center_ind[np.where(rej_ind_final==1)]
    pt_image = show_point_on_image(input_im,acc_pt,color=(0,1,0),convert_gray=convert_gray)
    pt_image = show_point_on_image(pt_image,rej_pt,color=(1,0,0),convert_gray=False)
    print(f'=======> saving image to {out_image_file}')
    image.imsave(out_image_file, pt_image)

def show_boxes_on_image(input,boxes,color = [0.64,0.27,0.61],convert_gray=True):
    b_image = np.zeros((np.shape(input)[0],np.shape(input)[1],3))
    if convert_gray:
        for i in range(0,3):
            b_image[:,:,i] = (input[:,:,0]+input[:,:,1]+input[:,:,2])/3
    else:
        for i in range(0,3):
            b_image[:,:,i] = input[:,:,i]

    for b in boxes:
        b_image[b[0]:b[2], b[1]] = color
        b_image[b[0]:b[2], b[3]] = color
        b_image[b[0], b[1]:b[3]] = color
        b_image[b[2], b[1]:b[3]] = color

    return b_image

def show_acc_rej_boxes_on_image(input_im,acc_boxes,rej_boxes,out_image_file,convert_gray=True):
    b_image = show_boxes_on_image(input_im,acc_boxes,color=(0,1,0),convert_gray=convert_gray)
    b_image = show_boxes_on_image(b_image,rej_boxes,color=(1,0,0),convert_gray=False)
    print(f'=======> saving image to {out_image_file}')
    image.imsave(out_image_file, b_image)
    return b_image

def get_gt_data(name):
    [im_size,_,_,gt_image_name,_,_,gt_color,patch_sz,dir] = get_image_info(name)
    gt_image = np.array(Image.open(gt_image_name.split('.')[0]+"_gt."+gt_image_name.split('.')[1]))
    gt_image = gt_image[:,:,:3]
    print(gt_image_name.split('.')[0]+"_gt."+gt_image_name.split('.')[1])

    substring = "Cell"
    if substring in name: 
        row_gt, col_gt = get_gt_data_bbx_csv(gt_image_name)
    else:
        row_gt, col_gt = get_gt_data_image(gt_image,gt_color)


    print("---------------create original no padding gt image to show row_gt, col_gt ---------------")
    print(f'row_gt shape: {row_gt.shape}, col_gt shape: {col_gt.shape}, gt_image.shape={gt_image.shape}')
    gt_markup_image = show_point_on_image(gt_image,zip(np.int32(row_gt),np.int32(col_gt)))
    gt_markup_image_file = gt_image_name.split('.')[0]+'gt_markup_image_nopadding.png'
    print(f'=======> saving image to {gt_markup_image_file}')
    image.imsave(gt_markup_image_file, normalize_image(gt_markup_image))

    ratio = [im_size[0]/np.float(gt_image.shape[0]),im_size[1]/np.float(gt_image.shape[1])]
    patch_sz_hf = patch_sz//2
    # row_gt, col_gt represent gt box centers after padding
    row_gt = np.int32(row_gt*ratio[0]) + patch_sz_hf
    col_gt = np.int32(col_gt*ratio[1]) + patch_sz_hf
    print(f'loading {len(row_gt)} ground truth points from {gt_image_name}, \
        im_size={im_size},gt_image.shape={gt_image.shape}, patch_sz_hf={patch_sz_hf}, ratio={ratio}')
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

    col_gt,row_gt=[],[]

    xx = group_data[['coords_x']].to_numpy()
    yy = group_data[['coords_y']].to_numpy()
    
    for x,y in zip(xx,yy):

        coords_x=np.array([int(i) for i in x[0].split(',')][0:-1])
        m_x=np.median(coords_x) if len(set(coords_x))>2 else np.mean(coords_x) 
        col_gt.append(m_x)

        coords_y=np.array([int(i) for i in y[0].split(',')][0:-1])
        m_y=np.median(coords_y) if len(set(coords_y))>2 else np.mean(coords_y) 
        row_gt.append(m_y)

        print(f'corrds_x={coords_x},corrds_y={coords_y}, m_x={m_x}, m_y={m_y}')

    return np.array(row_gt), np.array(col_gt)

bbx_center=lambda x, y : ((np.min(x)+np.max(x))/2.0,(np.min(y)+np.max(y))/2.0)

def get_gt_data_bbx_csv(gt_image_name):
    gt_data = pd.read_csv(gt_image_name.split('.')[0]+"_gtdata.csv")
    group_data = gt_data.loc[gt_data["group"] == "lymphocyte"]

    col_gt,row_gt=[],[]

    xx = group_data[['coords_x']].to_numpy()
    yy = group_data[['coords_y']].to_numpy()
    
    for x,y in zip(xx,yy):

        coords_x=np.array([int(i) for i in x[0].split(',')])
        coords_y=np.array([int(i) for i in y[0].split(',')])
        m_x,m_y=bbx_center(coords_x,coords_y)
        col_gt.append(m_x)
        row_gt.append(m_y)

        print(f'corrds_x={coords_x},corrds_y={coords_y}, min_x={np.min(coords_x)},max_x={np.max(coords_x)},m_x={m_x}, m_y={m_y}')

    return np.array(row_gt), np.array(col_gt)

def get_gt_data_box_center_csv(gt_image_name): 
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
    r = 1.5*patch_sz
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

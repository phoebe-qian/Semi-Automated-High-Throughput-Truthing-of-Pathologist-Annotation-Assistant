#!/usr/bin/env python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from aux_funcs import *
from count_gt_vs_output import *
import os
import time
import matplotlib.image as image
from sklearn.neighbors import NearestNeighbors
import scipy.misc
import cv2
from skimage.feature import match_template
from conf import get_image_info
from count_gt_vs_output import get_gt_data
import datetime
import random
import pandas as pd
from sklearn.metrics import accuracy_score

global boxes_rej_to_user
global boxes_acc_to_user
global boxes_id_rej_to_user
global boxes_id_acc_to_user
global user_rej
global user_acc

global start_loc
global end_loc

global to_show
global fig2
global clicks_counter
global Iteration
global Predicted_Count
global Accuracy 
global Time_Duration 
global Ground_Truth_Count 


Iteration = []
Predicted_Count = []
Accuracy = []
Time_Duration =[]
Ground_Truth_Count = []
dens_sTILS = []
precision = []

clicks_counter = 0

boxes_rej_to_user = []
boxes_acc_to_user = []
boxes_id_rej_to_user = []
boxes_id_acc_to_user = []
user_rej = []
user_acc = []
clicks_hist =[]

training_iters = 20
batch_size = 1

filt_mag = 0.1
bias_mag = 0.0


"""
def find_acc(gt_count, our_count):
    errorRate = (abs(gt_count-our_count)/gt_count)*100
    acc = 100 - errorRate
    return acc
"""
def find_prec (false_pos, true_pos):
    prec = (true_pos/(true_pos+false_pos))*100
    return prec

def find_acc(gt_count, true_pos):
    recall = (true_pos/gt_count)*100
    return recall

def find_density(our_count):
    sTILS_area = curr_patch_sz**2
    all_sTILS_area = our_count*sTILS_area
    density_s = (all_sTILS_area/ROIarea)*100
    return density_s


def get_patches(G, I, G_th,all_boxes,im_h,im_w,max_num_cells,iou_threshold=0.2):
    """
    Getting the potential locations of the repeating object based on the network outpout G
    """
    p = []
    scores = tf.reshape(G,[-1])
    all_scores = tf.stack(scores, 0)
    selected_indices = tf.image.non_max_suppression(all_boxes, scores, max_num_cells, iou_threshold=iou_threshold)
    selected_boxes = tf.gather(all_boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    I = tf.slice(I, [0, patch_sz_hf, patch_sz_hf, 0], [1, im_h , im_w , 1])

    pos_ind = tf.where(selected_scores > G_th)

    return [selected_indices,pos_ind,all_scores,selected_scores]

def unpool(I,loc,shape):
    """
    Tensorflow unpooling using locations from max-pooling operator.
    """
    bs = int(I.shape[0])

    U = []
    for b in range(bs):
        flatten_loc = tf.reshape(loc[b], [-1,1])
        flatten_val = tf.reshape(I[b], [-1])

        c_shape = tf.constant([shape[1]*shape[2]*shape[3]], dtype=flatten_loc.dtype)
        ups = tf.scatter_nd(flatten_loc, flatten_val, c_shape)
        U.append(tf.reshape(ups, shape[1:]))

    return tf.stack(U,axis=0)

def network(x, enc_w1, enc_b1, enc_w2, enc_b2, enc_w3, enc_b3, enc_w4, enc_b4, im_h,im_w,nf1,nf2):
    """
    Forward pass. The network architecture is explained in the paper.
    """
    #layer 1 (encoding layer)
    k = 1
    x = tf.slice(x, [0, patch_sz_hf, patch_sz_hf, 0], [1, im_h , im_w , 3])
    x = tf.nn.conv2d(x, enc_w1,strides=[1, k, k, 1], padding='SAME')
    x = tf.nn.bias_add(x, enc_b1)
    x = tf.nn.relu(x)
    
    #layer 2 (encoding layer)
    x = tf.nn.conv2d(x, enc_w2, strides=[1, k, k, 1], padding='SAME')
    _, loc1 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    loc1 = tf.stop_gradient(loc1)
    loc1 = tf.cast(loc1, tf.int32)
    #2by2 max pooling operation
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    x = tf.nn.bias_add(x, enc_b2)
    x_smaller = x

    # Keep the features for the loss function
    features = tf.reshape(x_smaller,[-1,nf2])
    features = tf.transpose(tf.transpose(features)/tf.sqrt(tf.reduce_sum(tf.square(features),axis=1)))
    features = tf.where(tf.is_nan(features),tf.zeros(tf.shape(features)),features)
    features = tf.reshape(features,(1,tf.cast(im_h/2,tf.int32),tf.cast(im_w/2,tf.int32),nf2))
    x = tf.nn.relu(features)
    
    #layer 3 (decoding layer)
    #2by2 unpooling step that produce the network's output classification map, C9x) to have the same resolution
    #as the input image
    x = unpool(x,loc1,[1,im_h,im_w,nf2])
    x = tf.nn.conv2d_transpose(x, enc_w3, [1,im_h,im_w,nf1], [1, 1, 1, 1], 'SAME')
    x = tf.nn.bias_add(x, enc_b3)
    x = tf.nn.relu(x)

    #layer 4
    x = tf.nn.conv2d(x, enc_w4,strides=[1, k, k, 1], padding='SAME')
    x = tf.nn.bias_add(x, enc_b4)

    return [x,tf.reshape(features,[-1,nf2])]

def cost_G_obj(all_scores,features,rej_final, acc_final,rej_final_smaller,acc_final_smaller,im_h,im_w):
    """
    Calculate the loss (L_label+L_sub)
    """
    rej_boxes_final = tf.cast(tf.where(tf.not_equal(rej_final, 0)), tf.int32)
    w_rej_final = tf.gather(all_scores, rej_boxes_final)

    acc_boxes_final = tf.cast(tf.where(tf.equal(acc_final, 1)), tf.int32)
    w_acc_final = tf.gather(all_scores, acc_boxes_final)

    w_mult_v_pos = tf.reduce_mean(tf.pow(w_acc_final - 2, 2))
    w_mult_v_neg = tf.reduce_mean(tf.pow(w_rej_final + 2, 2))

    rej_boxes_final_smaller = tf.cast(tf.where(tf.not_equal(rej_final_smaller, 0)), tf.int32)
    acc_boxes_final_smaller = tf.cast(tf.where(tf.equal(acc_final_smaller, 1)), tf.int32)

    rej_features = tf.gather(features,rej_boxes_final_smaller)
    acc_features = tf.gather(features,acc_boxes_final_smaller)

    n = tf.cast(nf2/2,tf.int32)
    cost = tf.reduce_mean(w_mult_v_pos + w_mult_v_neg) + 1*tf.reduce_mean(tf.pow(rej_features[:,:,0:n],2)) + 1*tf.reduce_mean(tf.pow(acc_features[:,:,n:],2))

    return cost

def is_inside_box(p,box):
    "return true if point p (y,x) is inside a box (y1x1y2x2)"
    y,x = p 
    inside = y >= box[0] and y <= box[2] and x >= box[1] and x <= box[3]
    return inside

def get_feedback_from_gt(row_gt, col_gt):

    global boxes_rej_to_user
    global boxes_acc_to_user
    global boxes_id_rej_to_user
    global boxes_id_acc_to_user
    global user_rej
    global user_acc
    global clicks_counter
    global patch_sz_hf

    pq = patch_sz_hf//2
    print(f'receiving feedback from gt data ...: ')
    print(f'# of boxes_rej_to_user: {len(boxes_rej_to_user)},{boxes_rej_to_user}')
    print(f'# of boxes_acc_to_user: {len(boxes_acc_to_user)}, {boxes_acc_to_user}')

    #row_gt, col_gt = get_gt_data(image_name)

    for i, b in enumerate(boxes_rej_to_user):
        # if the click is indie a negative window - mark it in black and add it to the positive bucket labels
        b_=[b[0]+pq,b[1]+pq,b[2]-pq,b[3]-pq]
        for p in zip(row_gt,col_gt):
            if is_inside_box(p,b_):
                user_acc.append(boxes_id_rej_to_user[i])
                """
                [boxes_rej_to_user[i][0]:boxes_rej_to_user[i][2], boxes_rej_to_user[i][1], :] = [0, 0, 0]
                [boxes_rej_to_user[i][0]:boxes_rej_to_user[i][2], boxes_rej_to_user[i][3], :] = [0, 0, 0]
                [boxes_rej_to_user[i][0], boxes_rej_to_user[i][1]:boxes_rej_to_user[i][3], :] = [0, 0, 0]
                [boxes_rej_to_user[i][2], boxes_rej_to_user[i][1]:boxes_rej_to_user[i][3], :] = [0, 0, 0]
                """
                clicks_counter += 1
                break

    for i,b in enumerate(boxes_acc_to_user):
        # if the click is indie a positive window - mark it in black and add it to the negative bucket labels
        found_b = False
        b_=[b[0]+pq,b[1]+pq,b[2]-pq,b[3]-pq]
        for p in zip(row_gt, col_gt):
            if is_inside_box(p,b_):
                found_b = True
                break
        if not found_b: 
            user_rej.append(boxes_id_acc_to_user[i])
            """
            [boxes_acc_to_user[i][0]:boxes_acc_to_user[i][2], boxes_acc_to_user[i][1], :] = [0, 0, 0]
            [boxes_acc_to_user[i][0]:boxes_acc_to_user[i][2], boxes_acc_to_user[i][3], :] = [0, 0, 0]
            [boxes_acc_to_user[i][0], boxes_acc_to_user[i][1]:boxes_acc_to_user[i][3], :] = [0, 0, 0]
            [boxes_acc_to_user[i][2], boxes_acc_to_user[i][1]:boxes_acc_to_user[i][3], :] = [0, 0, 0]
            """
            clicks_counter += 1


"""
def show_to_user(input, boxes_rej, boxes_acc, acc_map, center_rej, center_acc,step):
  
Present the negative and the positive queries to the user

    global to_show
    global plt_to_user
    global fig2
    global IS_EXP_MODE
    global to_show_offline

    dims = np.shape(input)
    to_show = np.zeros((dims[1], dims[2], 3))
    to_show = input[0,:,:,:].copy()
    for i in range(0, len(boxes_rej)):
        to_show[boxes_rej[i][0]:boxes_rej[i][2], boxes_rej[i][1], :] = [1, 0, 0] #[0.33, 0, 0]
        to_show[boxes_rej[i][0]:boxes_rej[i][2], boxes_rej[i][1]+1, :] = [1, 0, 0] #[0.33, 0, 0]
        to_show[boxes_rej[i][0]:boxes_rej[i][2], boxes_rej[i][3]-1, :] = [1, 0, 0] #[0.33, 0, 0]

        to_show[boxes_rej[i][0], boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0] #[0.33, 0, 0]
        to_show[boxes_rej[i][0]+1, boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0] #[0.33, 0, 0]
        to_show[boxes_rej[i][2], boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0] #[0.33, 0, 0]
        to_show[boxes_rej[i][2]-1, boxes_rej[i][1]:boxes_rej[i][3], :] = [1, 0, 0] #[0.33, 0, 0]

        to_show[center_rej[i][0], center_rej[i][1], :] = [0, 0, 0] #[0.33, 0, 0]

    for i in range(0, len(boxes_acc)):
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][1], :] = [0, 1, 0] #[0, 0.33, 0]
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][1]+1, :] = [0, 1, 0] #[0, 0.33, 0]
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][3], :] = [0, 1, 0] #[0, 0.33, 0]
        to_show[boxes_acc[i][0]:boxes_acc[i][2], boxes_acc[i][3]-1, :] = [0, 1, 0] #[0, 0.33, 0]

        to_show[boxes_acc[i][0], boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0] #[0, 0.33, 0]
        to_show[boxes_acc[i][0]+1, boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0] #[0, 0.33, 0]
        to_show[boxes_acc[i][2], boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0] #[0, 0.33, 0]
        to_show[boxes_acc[i][2]-1, boxes_acc[i][1]:boxes_acc[i][3], :] = [0, 1, 0] #[0, 0.33, 0]

        to_show[center_acc[i][0], center_acc[i][1], :] = [0, 0, 0] #[0.33, 0, 0]

    plt.close(1)
    print('starting UI session')
    fig2 = plt.figure(2, figsize=(80, 60))
    to_show_offline = to_show.copy()
    fig2.canvas.mpl_connect('button_press_event', onclick_userCorrection)

    plt_to_user = plt.imshow(to_show)
    plt.show()

def onclick_test(event):
    [x, y] = event.xdata, event.ydata
    print(f'click received: {x,y}')
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    
def onclick_userCorrection(event):

    Event when the user clicks on the image. In case it is inside a negative or a positive window, this window
    is changed to have positive ot negative label respectively.
    In this version we removed the option for 'unchanged' button. It can be integrated easily.

    global boxes_rej_to_user
    global boxes_acc_to_user
    global boxes_id_rej_to_user
    global boxes_id_acc_to_user
    global user_rej
    global user_acc
    global to_show
    global plt_to_user
    global fig2
    global clicks_counter
    global step

    [x, y] = event.xdata, event.ydata
    for i in range(0, len(boxes_rej_to_user)):
        # if the click is indie a negative window - mark it in black and add it to the positive bucket labels
        if y > boxes_rej_to_user[i][0] and y < boxes_rej_to_user[i][2] and x > boxes_rej_to_user[i][1] and x < \
                boxes_rej_to_user[i][3]:
            user_acc.append(boxes_id_rej_to_user[i])
            to_show[boxes_rej_to_user[i][0]:boxes_rej_to_user[i][2], boxes_rej_to_user[i][1], :] = [0, 0, 0]
            to_show[boxes_rej_to_user[i][0]:boxes_rej_to_user[i][2], boxes_rej_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_rej_to_user[i][0], boxes_rej_to_user[i][1]:boxes_rej_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_rej_to_user[i][2], boxes_rej_to_user[i][1]:boxes_rej_to_user[i][3], :] = [0, 0, 0]
            plt_to_user.set_data(to_show)
            clicks_counter += 1
            break

    for i in range(0, len(boxes_acc_to_user)):
        # if the click is indie a positive window - mark it in black and add it to the negative bucket labels
        if y > boxes_acc_to_user[i][0] and y < boxes_acc_to_user[i][2] and x > boxes_acc_to_user[i][1] and x < \
                boxes_acc_to_user[i][3]:
            user_rej.append(boxes_id_acc_to_user[i])
            to_show[boxes_acc_to_user[i][0]:boxes_acc_to_user[i][2], boxes_acc_to_user[i][1], :] = [0, 0, 0]
            to_show[boxes_acc_to_user[i][0]:boxes_acc_to_user[i][2], boxes_acc_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_acc_to_user[i][0], boxes_acc_to_user[i][1]:boxes_acc_to_user[i][3], :] = [0, 0, 0]
            to_show[boxes_acc_to_user[i][2], boxes_acc_to_user[i][1]:boxes_acc_to_user[i][3], :] = [0, 0, 0]
            plt_to_user.set_data(to_show)
            clicks_counter += 1
            break
    fig2.canvas.draw()
    fig2.canvas.flush_events()
"""
"""
def should_stop(rej_final, acc_final, all_scores):
    
    Stopping condition. If all the locations of the negative labels received a low
    value and the locations of the positive labels high value (in the network output).
    """
"""
    if np.sum(all_scores[np.where(rej_final != 0)] >= -0.4) != 0:
        return False

    if np.sum(all_scores[np.where(acc_final == 1)] <= 0.9) != 0:
        return False

    return True
"""
def continue_training(rej_final, acc_final, all_scores, display=False, neg_thld=-0.4, pos_thld=0.9):
    """
    Stopping condition. If all the locations of the negative labels received a low
    value and the locations of the positive labels high value (in the network output).
    """
    neg_ind = np.where(rej_final != 0)
    neg_max = np.max(all_scores[neg_ind])
    pos_ind = np.where(acc_final == 1)
    pos_min = np.min(all_scores[pos_ind])

    if display:
        print(f'{len(neg_ind)} neg lables, neg_max={neg_max}, neg_thld={neg_thld}, {len(pos_ind)} pos labels, pos_min={pos_min}, pos_thld={pos_thld}')
    
    if neg_max < neg_thld and pos_min > pos_thld:
        return False

    return True

def calc_cost_AE_prep(x_corr,th,all_boxes,im_h,im_w,iou_threshold=0.2):
    """
    Calculate the initilize positive and negative buckets using normalized cross correlation.
    """

    scores_x_corr = tf.reshape(tf.slice(
                                x_corr, 
                                [0, patch_sz_hf, patch_sz_hf, 0], 
                                [1, im_h, im_w, 1]),
                                [-1])
    ind = list(range((im_h ) * (im_w )))

    selected_indices_pos = tf.image.non_max_suppression(all_boxes, scores_x_corr,im_h*im_w, iou_threshold=iou_threshold)
    selected_x_scores_pos = tf.gather(scores_x_corr, selected_indices_pos)
    selected_center_ind_pos = tf.gather(all_center_ind, selected_indices_pos)
    selected_ind_pos = tf.gather(ind, selected_indices_pos)
    pos = tf.cast(tf.where(selected_x_scores_pos > th), tf.int32)
    ind_pos = tf.squeeze(tf.gather(selected_ind_pos, pos))
    to_show_pos = tf.squeeze(tf.gather(selected_center_ind_pos, pos))

    selected_indices_neg = tf.image.non_max_suppression(all_boxes, np.abs(scores_x_corr), im_h*im_w, iou_threshold=iou_threshold)
    selected_x_scores_neg = tf.gather(scores_x_corr, selected_indices_neg)
    selected_center_ind_neg = tf.gather(all_center_ind, selected_indices_neg)
    selected_ind_neg = tf.gather(ind, selected_indices_neg)
    neg = tf.cast(tf.where(selected_x_scores_neg < -th*.75), tf.int32)
    ind_neg = tf.squeeze(tf.gather(selected_ind_neg, neg))
    to_show_neg = tf.squeeze(tf.gather(selected_center_ind_neg, neg))

    return [to_show_pos, to_show_neg, ind_pos, ind_neg]

def adding_rej(path,im_h,im_w,rej_ind_final):
    """
    Adding negative points in borders
    """
    tmp_rej = np.reshape(rej_ind_final,(im_h,im_w))
    tmp_rej[0:2,:] = 1
    tmp_rej[:,0:2] = 1
    tmp_rej[-2:,:] = 1
    tmp_rej[:,-2:] = 1
    rej_ind_final = np.reshape(tmp_rej,[-1])
    return rej_ind_final

def calc_smaller_ind(curr_pos_ind,im_h,im_w,factor,output_mat=False):
    """
    Mapping between the image presented to the user (which is bigger by a factor of 2) and the
    real size of the image.
    """
    mat = np.zeros((im_h,im_w))
    mat_smaller = np.zeros((np.int32(im_h/factor),np.int32(im_w/factor)))

    [row,col] = np.unravel_index(curr_pos_ind, [im_h, im_w])
    row_small = np.int32(row/factor)
    col_small = np.int32(col/factor)
    for i in range(0,len(row_small)):
        mat_smaller[row_small[i],col_small[i]] = 1
    if output_mat:
        return np.reshape(mat_smaller,-1)
    else:
        return np.where(np.reshape(mat_smaller == 1,-1))[0]

def calc_larger_ind(curr_pos_ind,im_h,im_w,factor):
    """
    Mapping between the input image and the image presnted to the user (which is bigger
    by a factor of 2).
    """
    mat = np.zeros((im_h,im_w))
    mat_larger = np.zeros((im_h*factor,im_w*factor))

    [row,col] = np.unravel_index(curr_pos_ind, [im_h, im_w])
    row_large = row*factor
    col_large = col*factor
    for i in range(0,len(row_large)):
        mat_larger[row_large[i],col_large[i]] = 1
    return np.where(np.reshape(mat_larger == 1,-1))[0]

def change_to_closest_to_ask(curr_ind,ind_to_ask,im_h,im_w):
    """
    Change curr_ind to the closest index from ind_to_ask
    """
    [row_curr,col_curr] = np.unravel_index(curr_ind, [im_h, im_w])
    [row_ind_to_ask,col_ind_to_ask] = np.unravel_index(ind_to_ask, [im_h, im_w])
    new_row = []
    new_col = []
    for i in range(0,len(row_curr)):
        d = []
        for j in range(0,len(row_ind_to_ask)):
            d.append(np.power(row_curr[i] - row_ind_to_ask[j],2) + np.power(col_curr[i] - col_ind_to_ask[j],2))
        new_row.append(row_ind_to_ask[np.argmin(d)])
        new_col.append(col_ind_to_ask[np.argmin(d)])

    mat = np.zeros((im_h,im_w))
    for i in range(0,len(new_row)):
        mat[new_row[i],new_col[i]] = 1
    return np.where(np.reshape(mat,-1) == 1)[0]


def main(dir,base_dir,im_size,image_name,number_of_patches,th,filename,gt_color,show_gray,max_num_cells,participant_name):
    init_time = time.time()
    global boxes_rej_to_user
    global boxes_acc_to_user
    global boxes_id_rej_to_user
    global boxes_id_acc_to_user
    global user_rej
    global user_acc
    global step

    global start_loc
    global end_loc
    global all_boxes
    global all_center_ind
    global nf1
    global nf2
    global nf3
    global nf1_up
    global nf2_up
    global patch_sz_hf
    global row_gt, col_gt
    global dataroot
    global ROIarea
    global curr_patch_sz

    row_gt, col_gt, ratio = get_gt_data(image_name)

    nf1 = number_of_patches
    nf2 = 2*nf1

    num_pool = 2
    factor = 2
    fil_sz_1 = 11
    fil_sz_2 = 7
    fil_sz_3 = 5
    fil_sz_4 = 5

    im_h = im_size[0]
    im_w = im_size[1]
    ROIarea = im_h*im_w

    # tf Graph input
    x_orig_tf = tf.get_variable(shape=(1, im_h + patch_sz_hf*2, im_w + patch_sz_hf*2, 3), name='x_orig', trainable=False)
    x_orig_ph = tf.placeholder(tf.float32, shape=[1, im_h + patch_sz_hf*2, im_w + patch_sz_hf*2, 3])
    assign_x_orig = tf.assign(x_orig_tf, x_orig_ph, validate_shape=False)

    x_corr = tf.placeholder(tf.float32, [None, im_h + patch_sz_hf*2, im_w + patch_sz_hf*2, 1])

    y = np.int32(np.linspace(0, im_h - 1, im_h )) # vertical
    x = np.int32(np.linspace(0, im_w - 1, im_w )) # horizontal 
    [i, j] = np.meshgrid(x, y)
    all_boxes = np.zeros(((im_h ) * (im_w ), 4), np.int32)
    all_boxes[:, 0] = np.reshape(j, -1)
    all_boxes[:, 1] = np.reshape(i, -1)
    all_boxes[:, 2] = np.reshape(j, -1) + patch_sz -1
    all_boxes[:, 3] = np.reshape(i, -1) + patch_sz -1

    all_center_ind = np.zeros(((im_h ) * (im_w), 2), np.int32)
    all_center_ind[:, 0] = np.reshape(j, -1) + patch_sz_hf # rows
    all_center_ind[:, 1] = np.reshape(i, -1) + patch_sz_hf # cols

    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        weights_generator = {
            'encoder_g1': tf.get_variable("encoder_g1",initializer = tf.constant(np.multiply(np.random.normal(size=[fil_sz_1, fil_sz_1, 3, nf1]), filt_mag),dtype=tf.float32)),
            'encoder_g2': tf.get_variable("encoder_g2",initializer = tf.constant(np.multiply(np.random.normal(size=[fil_sz_2, fil_sz_2, nf1, nf2]), filt_mag),dtype=tf.float32)),
            'encoder_g3': tf.get_variable("encoder_g3",initializer = tf.constant(np.multiply(np.random.normal(size=[fil_sz_3, fil_sz_3, nf1, nf2]), filt_mag),dtype=tf.float32)),
            'encoder_g4': tf.get_variable("encoder_g4",initializer = tf.constant(np.multiply(np.random.normal(size=[fil_sz_4, fil_sz_4, nf1, 1]), filt_mag),dtype=tf.float32)),
        }

        biases_generator = {
            'encoder_b1': tf.get_variable("encoder_b1",initializer = tf.constant(np.multiply(np.ones([nf1]), bias_mag),dtype=tf.float32)),
            'encoder_b2': tf.get_variable("encoder_b2",initializer = tf.constant(np.multiply(np.ones([nf2]), bias_mag),dtype=tf.float32)),
            'encoder_b3': tf.get_variable("encoder_b3",initializer = tf.constant(np.multiply(np.ones([nf1]), bias_mag),dtype=tf.float32)),
            'encoder_b4': tf.get_variable("eecoder_b4",initializer = tf.constant(np.multiply(np.ones([1]), bias_mag),dtype=tf.float32)),
        }

    output_dir = dir + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    learning_rate_G = 0.001
    learning_rate_G_init = 0.001
    patch_iou_threshold = 0.2 #representing the threshold for deciding whether boxes overlap too much with respect to IOU.

    print(f'preparing variabls ...')
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        # Graph declaration. Declare variables, the relative placeholders and the assign function
        scalar_to_separation_tf = tf.get_variable(shape=(), name='scalar_to_separation', trainable=False)
        scalar_to_separation_ph = tf.placeholder(tf.float32, shape=[])
        assign_scalar_to_separation = tf.assign(scalar_to_separation_tf, scalar_to_separation_ph, validate_shape=False)

        regularization_G_tf = tf.get_variable(shape=(), name='regularization', trainable=False)
        regularization_G_ph = tf.placeholder(tf.float32, shape=[])
        assign_regularization_G = tf.assign(regularization_G_tf, regularization_G_ph, validate_shape=False)

        num = (im_h ) * (im_w )
        rej_box_ind_final_tf = tf.get_variable(shape=(num,), name='rej_ind', trainable=False,initializer=tf.zeros_initializer())
        rej_box_ind_final_ph = tf.placeholder(tf.float32, shape=[num, ])
        assign_rej_ind_final = tf.assign(rej_box_ind_final_tf, rej_box_ind_final_ph, validate_shape=False)

        acc_box_ind_final_tf = tf.get_variable(shape=(num,), name='acc_ind', trainable=False,initializer=tf.zeros_initializer())
        acc_box_ind_final_ph = tf.placeholder(tf.float32, shape=[num, ])
        assign_acc_ind_final = tf.assign(acc_box_ind_final_tf, acc_box_ind_final_ph, validate_shape=False)

        num_smaller = (im_h/factor ) * (im_w/factor )
        rej_box_ind_final_smaller_tf = tf.get_variable(shape=(num_smaller,), name='rej_ind_smaller', trainable=False,initializer=tf.zeros_initializer())
        rej_box_ind_final_smaller_ph = tf.placeholder(tf.float32, shape=[num_smaller, ])
        assign_rej_ind_smaller_final = tf.assign(rej_box_ind_final_smaller_tf, rej_box_ind_final_smaller_ph, validate_shape=False)

        acc_box_ind_final_smaller_tf = tf.get_variable(shape=(num_smaller,), name='acc_ind_smaller', trainable=False,initializer=tf.zeros_initializer())
        acc_box_ind_final_smaller_ph = tf.placeholder(tf.float32, shape=[num_smaller, ])
        assign_acc_ind_smaller_final = tf.assign(acc_box_ind_final_smaller_tf, acc_box_ind_final_smaller_ph, validate_shape=False)

        th_tf = tf.get_variable(shape=(), name='th', trainable=False, initializer=tf.zeros_initializer())
        th_ph = tf.placeholder(tf.float32, shape=[])
        assign_th = tf.assign(th_tf, th_ph, validate_shape=False)


    ################################### network ###################################
    print(f'preparing classifer network ...')
    [G,features] = network(x_orig_tf, weights_generator['encoder_g1'], biases_generator['encoder_b1'],
                   weights_generator['encoder_g2'], biases_generator['encoder_b2'], weights_generator['encoder_g3'],
                   biases_generator['encoder_b3'], weights_generator['encoder_g4'], biases_generator['encoder_b4'],
                   im_h,im_w,nf1,nf2)

    print(f'get patches ...')
    [ind_box,pos_ind,all_scores,selected_scores] = get_patches(G, x_orig_tf, th_tf ,all_boxes,im_h,im_w,max_num_cells,iou_threshold=patch_iou_threshold)

    ################################### AE ###################################
    print(f'preparing AE ...')
    [show_pos_AE, show_neg_AE, x_corr_pos, x_corr_rej] = calc_cost_AE_prep(x_corr,th,all_boxes,im_h,im_w,iou_threshold=patch_iou_threshold)

    print(f'cost G object ...')
    cost_G_rejacc_maps = cost_G_obj(all_scores,features,rej_box_ind_final_tf,acc_box_ind_final_tf,rej_box_ind_final_smaller_tf,acc_box_ind_final_smaller_tf,im_h,im_w)


    all_vars = tf.global_variables()
    AE_G_vars = [v for v in all_vars if ('encoder' in v.name or 'decoder' in v.name)]
    lossl2_G = tf.add_n([tf.nn.l2_loss(v) for v in all_vars if('encoder' in v.name or 'decoder' in v.name)]) * regularization_G_tf  # _tmp=0.001
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        #optimizer for the network
        opt_G_rejacc_maps = tf.train.AdamOptimizer(learning_rate=learning_rate_G).minimize(cost_G_rejacc_maps + lossl2_G, var_list=AE_G_vars)
        #optimizer only for the initialization step
        opt_G_rejacc_maps_init = tf.train.AdamOptimizer(learning_rate=learning_rate_G_init).minimize(cost_G_rejacc_maps + lossl2_G, var_list=AE_G_vars)



    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    logdir = os.path.join(dataroot,"logs", datetime.datetime.now().strftime("%Y%m%d"))
    print(f'=======> tensorboard logging at {logdir}')



    # Launch the graph
    with tf.Session() as sess:
        #writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        # The size of these bonary maps are the size of the input image (vectorize)

        # The rejected map from the init. in order to ignore from showing in final rej map
        rej_ind_final_from_AE = np.zeros((((im_h ) * (im_w )),))
        acc_ind_final_from_AE = np.zeros((((im_h ) * (im_w )),))

        rej_ind_final = np.zeros((((im_h ) * (im_w )),))
        acc_ind_final = np.zeros((((im_h ) * (im_w )),))
        curr_ind_box = []

        sess.run(init)
        batch_x_orig = read_dataset(filename, im_h,im_w, patch_sz,show_gray)

        # there can be more than one window location
        # start_loc represent the box upper-left corner after padding
        window_loc = []
        for i in range(3):
            ind = random.randint(0, (len(row_gt)-1))
            locs = [col_gt[ind]-patch_sz_hf,row_gt[ind]-patch_sz_hf]
            window_loc.append(locs)
        if "34" in image_name:
            window_loc=[[108, 12], [266, 101], [89, 336]]
        print(f'window locs: {window_loc}')
        start_loc = window_loc
        """
        start_loc = []
        for w in window_loc:
            start_loc.append([int(w[0]*ratio[1]),int(w[1]*ratio[0])])
        """
        dims = batch_x_orig.shape
        img_corr = np.zeros((dims[1], dims[2],len(start_loc)))
        print(f'computing corss correlation btw input image with shape {dims} and the sample window ...')
        for i, (c,r) in enumerate(start_loc):
            print(f'cropping sample window size {patch_sz} at upper_left location (r,c)=({r},{c}) from input image with shape {dims} ')
            
            patch = batch_x_orig[0,r:r+patch_sz,c:c+patch_sz, :]
            res = np.zeros((dims[1]-patch_sz_hf*2, dims[2]-patch_sz_hf*2))
            print(f'i={i},res.shape={res.shape},patch.shape={patch.shape},batch_x_orig.shape={batch_x_orig.shape}')
            #calculate the normalize cross correlation between the input image and the initialized repeating object window
            for ch in range(0, 3):
                res = res + match_template(batch_x_orig[0,:, :, ch],patch[:, :, ch])

            res = res / 3
            img_corr[patch_sz_hf:-patch_sz_hf,patch_sz_hf:-patch_sz_hf,i] = res
            image.imsave(dir+'patch_%d.png'%(i), normalize_image(patch))
            image.imsave(dir+'res_%d.png'%(i), normalize_image(res))
            print(f'res_{i} min={np.min(res)}, max={np.max(res)}')
        image.imsave(dir+'img_input.png', normalize_image(batch_x_orig[0]))
        #image.imsave(dir+'img_corr.png', normalize_image(np.repeat(img_corr,3,axis=2)))

        #plt.show()

        for i in range(0,len(start_loc)):
            batch_x_corr = np.zeros((1, im_h + patch_sz_hf*2, im_w + patch_sz_hf*2, 1))
            batch_x_corr[0,:,:,0] = img_corr[:,:,i]

            # extract initialized positive and negative buckets from the normalized-correlation image
            x_corr_pos_curr = sess.run(x_corr_pos,feed_dict={x_corr: batch_x_corr})
            x_corr_rej_curr = sess.run(x_corr_rej,feed_dict={x_corr: batch_x_corr})

            print(f'=======> x_corr_pos_curr shape: {x_corr_pos_curr.shape}, x_corr_rej_curr shape:{x_corr_rej_curr.shape} ')
            
            rej_ind_final_from_AE[x_corr_rej_curr] = 1
            acc_ind_final_from_AE[x_corr_pos_curr] = 1

            # each being a hot vector in the size of im_h \times im_w with active entries set to 1 and inactive to 0
            rej_ind_final[x_corr_rej_curr] = 1
            acc_ind_final[x_corr_pos_curr] = 1

            #adding rej points in borders
            rej_ind_final = adding_rej(path,im_h,im_w,rej_ind_final)

            sess.run(assign_rej_ind_final, {rej_box_ind_final_ph: rej_ind_final})
            sess.run(assign_acc_ind_final, {acc_box_ind_final_ph: acc_ind_final})

            sess.run(assign_rej_ind_smaller_final, {rej_box_ind_final_smaller_ph: calc_smaller_ind(np.where(rej_ind_final==1)[0],im_h,im_w,factor,True)})
            sess.run(assign_acc_ind_smaller_final, {acc_box_ind_final_smaller_ph: calc_smaller_ind(np.where(acc_ind_final==1)[0],im_h,im_w,factor,True)})

        print("---------------create markup image to show initial acc (green) and rej (red) ---------------")
        show_acc_rej_on_image(batch_x_orig[0],all_center_ind,acc_ind_final,rej_ind_final,dir+'init_gt_image.png')


        sess.run(assign_x_orig, {x_orig_ph: batch_x_orig})

        regularization_G = 0
        sess.run(assign_regularization_G, {regularization_G_ph: regularization_G})
        G_th = 0.5
        sess.run(assign_th, {th_ph: G_th})
        # Initialize step: train the network with the initialize labels
        print(f'Initialize step: train the network with the initialize labels ...')
        i = 0
        curr_all_scores = sess.run(all_scores)
        while continue_training(rej_ind_final, acc_ind_final, curr_all_scores):
            #print(f'updating models: iteration {i}')
            i += 1
            sess.run(opt_G_rejacc_maps_init)
            curr_all_scores = sess.run(all_scores)

        print(f'finished updating models after iteration {i}')
        continue_training(rej_ind_final, acc_ind_final, curr_all_scores, display=True)

        curr_G_save = sess.run(G)
        print(f'Initialize step done')

        print(f'model inference using the initial weights ...')
        input = sess.run(x_orig_tf)

        regularization_G = 0.001
        sess.run(assign_regularization_G, {regularization_G_ph: regularization_G})
        G_th = 0.5
        sess.run(assign_th, {th_ph: G_th})

        print(f'going through the iterations ...')
        training_iters=100
        reviewed_ind=[]
        step = 0
        if step==0:
            print("---------------create gt image to show row_gt, col_gt ---------------")
            print(f'row_gt shape: {row_gt.shape}, col_gt shape: {col_gt.shape}')
            gt_markup_image = show_point_on_image(input[0],zip(row_gt,col_gt))
            print(f'=======> saving image to {dir}gt_markup_image'+'.png')
            image.imsave(dir+'gt_markup_image.png', normalize_image(gt_markup_image))

        while step * batch_size <= training_iters: 
            curr_features = sess.run(features)#features from CNN

            [curr_pos_ind,curr_ind_box,curr_all_scores] = sess.run([pos_ind, ind_box,all_scores])
            curr_center_ind = all_center_ind[curr_ind_box]
            curr_ind_box = curr_ind_box[0:curr_pos_ind.shape[0]]
            curr_ind_box_small = calc_smaller_ind(curr_pos_ind,im_h,im_w,factor)

            potential_map = np.zeros((im_h ) * (im_w ))
            potential_map[curr_ind_box[0:curr_pos_ind.shape[0]]] = 1
            potential_map_tmp = np.where(potential_map == 1)[0]
            potential_map = np.reshape(potential_map, (im_h , im_w ))
            ind_G = np.squeeze(np.where(potential_map == 1))

            acc_ind_final_new = np.reshape(acc_ind_final, (im_h , im_w ))
            ind_final = np.squeeze(np.where(acc_ind_final_new == 1))
            if ind_final.ndim == 1:
                ind_final = np.expand_dims(ind_final,1)
            if ind_G.ndim == 1:
                ind_G = np.expand_dims(ind_G,1)

            """
            P: Query sampling steps: Avoiding Redundant queries and obtaining informative queries
            POTENTIAL LOCATION OF REPEATING OBJECTS
            """
            
            # If the new candidiates fall very close to the ones there were allready labeled - ignore them
            ignore_from_show = []
            for i in range(0, ind_G.shape[1]):
                for j in range(0, ind_final.shape[1]):
                    if (ind_G[0][i] > ind_final[0][j] - patch_sz/3) \
                        and (ind_G[0][i] < ind_final[0][j] + patch_sz/3) \
                        and (ind_G[1][i] > ind_final[1][j] - patch_sz/3) \
                        and (ind_G[1][i] < ind_final[1][j] + patch_sz/3) \
                        and (ind_final[0][j] != ind_G[0][i] or ind_final[1][j] != ind_G[1][i]):

                        acc_ind_final_new[ind_final[0][j]][ind_final[1][j]] = 1
                        acc_ind_final_new[ind_G[0][i]][ind_G[1][i]] = 0
                        ignore_from_show.append(potential_map_tmp[i])
            
            acc_ind_final = np.reshape(acc_ind_final_new, [-1])

            curr_features_acc = curr_features[calc_smaller_ind(np.where(acc_ind_final == 1),im_h,im_w,factor)]
            curr_features_rej = curr_features[calc_smaller_ind(np.where(rej_ind_final == 1),im_h,im_w,factor)]
            # Concatenate the features that are related to positive and negative labeled windows
            tot_features = np.concatenate((curr_features_acc,curr_features_rej))
            tot_features_bool = np.concatenate((np.ones((curr_features_acc.shape[0])),-np.ones((curr_features_rej.shape[0]))))
            
            # P: Sample user queries - with positive label (Split potential locations into 2 groups: 
                # Wp based on distance to labeled pixels in P)
                # Calculate nearest neigbour in feature space
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tot_features)

            ind_to_ask = curr_ind_box
            ind_to_ask = np.setdiff1d(ind_to_ask,ignore_from_show)

            # remove index that are already labeled
            #ind_to_ask = np.setdiff1d(ind_to_ask, np.where((rej_ind_final + acc_ind_final) >= 1))
            ind_to_ask = np.setdiff1d(ind_to_ask, np.where((acc_ind_final) >= 1))
            ind_to_ask = np.setdiff1d(ind_to_ask, np.array(reviewed_ind))
            ind_to_ask_small = calc_smaller_ind(ind_to_ask,im_h,im_w,factor)

            # calculate the distance for each potential location
            nn=np.array([])
            if len(ind_to_ask_small) > 0:
                distances, indices = nbrs.kneighbors(curr_features[ind_to_ask_small])
                nn = tot_features_bool[indices]

            nn_pos = np.where(nn == 1)[0]
            nn_neg = np.where(nn == -1)[0]

            if "coke" in filename:
                number_to_show = 2
            else:
                number_to_show = 5

            max_clusters = number_to_show*2
            
            a = []
            if len(ind_to_ask_small[nn_pos]) > 0:
                # cluster the potential locations that are realted to the positive windows
                n_clusters = np.min((max_clusters,len(ind_to_ask_small[nn_pos])))
                kmeans_pos = KMeans(n_clusters=n_clusters, random_state=0).fit(curr_features[ind_to_ask_small[nn_pos]])
                a_candidates = []
                farer_dist_pos = np.argsort(distances[nn_pos,0])[::-1]
                pos_labels = np.unique(kmeans_pos.labels_)
                # for each cluster, pick the farthest candidate
                for i in range(0,len(pos_labels)):
                    worst_dist_ind = np.where(kmeans_pos.labels_[farer_dist_pos] == pos_labels[i])[0][0]
                    worst_dist = distances[nn_pos][farer_dist_pos[worst_dist_ind]]
                    a_candidates.append([worst_dist,worst_dist_ind])
                a_candidates = np.asarray(a_candidates)
                a = ind_to_ask_small[nn_pos][farer_dist_pos[np.int32(a_candidates[np.argsort(a_candidates[:,0])][-number_to_show:][:,1])]]


            # P: Sample user queries - with negative label (Split potential locations into 2 groups: Wn based on distance to labeled pixels in N)
            r = []
            if len(ind_to_ask_small[nn_neg]) > 0:
                # cluster the potential locations that are realted to the negative windows
                n_clusters = np.min((max_clusters,len(ind_to_ask_small[nn_neg])))
                kmeans_neg = KMeans(n_clusters=n_clusters, random_state=0).fit(curr_features[ind_to_ask_small[nn_neg]])
                r_candidates = []
                farer_dist_neg = np.argsort(distances[nn_neg,0])[::-1]
                neg_labels = np.unique(kmeans_neg.labels_)
                # for each cluster, pick the farthest candidate to avoid query windows which are similar to ones alredy labeled.
                for i in range(0,len(neg_labels)):
                    worst_dist_ind = np.where(kmeans_neg.labels_[farer_dist_neg] == neg_labels[i])[0][0]
                    worst_dist = distances[nn_neg][farer_dist_neg[worst_dist_ind]]
                    r_candidates.append([worst_dist,worst_dist_ind])
                r_candidates = np.asarray(r_candidates)
                r = ind_to_ask_small[nn_neg][farer_dist_neg[np.int32(r_candidates[np.argsort(r_candidates[:,0])][-number_to_show:][:,1])]]

            if len(a) > 0:
                a = calc_larger_ind(a,np.int32(im_h/factor),np.int32(im_w/factor),factor)
                a = change_to_closest_to_ask(a,ind_to_ask,im_h,im_w)
            if len(r) > 0:
                r = calc_larger_ind(r,np.int32(im_h/factor),np.int32(im_w/factor),factor)
                r = change_to_closest_to_ask(r,ind_to_ask,im_h,im_w)
            #'r' and 'a' are the locations to ask the user

            # evaluate the performance of the current model 
            final(row_gt,col_gt,dir,step,input,ind_to_ask,acc_ind_final,init_time,clicks_counter,patch_sz,im_size)
            # if all the potential locations have high score in the network's output map
            
            """
            if len(ind_to_ask) > 0:
                min_ask=np.min(curr_all_scores[ind_to_ask])
                print(f'=======> min at ind_to_ask={min_ask}')
                if min_ask >= 0.85: #stop the algorithm
                    break
            """

            #boxes_id_rej_to_user = np.int32(np.setdiff1d(r, np.where((rej_ind_final + acc_ind_final) >= 1)))
            #boxes_id_acc_to_user = np.int32(np.setdiff1d(a, np.where((acc_ind_final + rej_ind_final) >= 1)))
            print(f'formulating boxes_id_rej_to_user: r->{len(r)}, reviewed_ind->{len(reviewed_ind)}')
            print(f'formulating boxes_id_acc_to_user: a->{len(a)}, reviewed_ind->{len(reviewed_ind)}')
            boxes_id_rej_to_user = np.int32(np.setdiff1d(r, np.array(reviewed_ind)))
            boxes_id_acc_to_user = np.int32(np.setdiff1d(a, np.array(reviewed_ind)))
            reviewed_ind.extend(list(boxes_id_rej_to_user))
            reviewed_ind.extend(list(boxes_id_acc_to_user))

            # convert the queries locations to windows
            boxes_rej_to_user = all_boxes[boxes_id_rej_to_user]
            all_center_ind_rej = all_center_ind[boxes_id_rej_to_user]
            boxes_acc_to_user = all_boxes[boxes_id_acc_to_user]
            all_center_ind_acc = all_center_ind[boxes_id_acc_to_user]
            #plt.close()
            #plt.figure(1)

            # ask the user
            """
            if (boxes_id_rej_to_user.shape[0] > 0 or boxes_id_acc_to_user.shape[0] > 0):
                show_to_user(input, boxes_rej_to_user, boxes_acc_to_user, acc_ind_final, all_center_ind_rej, all_center_ind_acc, step)
            """

            get_feedback = boxes_id_rej_to_user.shape[0] > 0 or boxes_id_acc_to_user.shape[0] > 0
            if (get_feedback):
                get_feedback_from_gt(row_gt,col_gt)
                out_image_file=f'feedback_query_boxes_markup_step{step}.png'
                b_image=show_acc_rej_boxes_on_image(input[0],boxes_acc_to_user,boxes_rej_to_user,dir+out_image_file,convert_gray=True)

            print(f'step*batch_size={step*batch_size}, corrected lables: # of user_acc:{len(user_acc)}, # of user_rej:{len(user_rej)}')

            user_tot = np.concatenate((user_rej,user_acc), axis=0)
            user_rej = np.int32(np.concatenate((user_rej, np.setdiff1d(boxes_id_rej_to_user, user_tot)), axis=0))
            user_acc = np.int32(np.concatenate((user_acc, np.setdiff1d(boxes_id_acc_to_user, user_tot)), axis=0))
            print(f'step*batch_size={step*batch_size}, final labels: # of user_acc:{len(user_acc)}, # of user_rej:{len(user_rej)}')
            
            user_rej = np.setdiff1d(user_rej, -1)
            user_acc = np.setdiff1d(user_acc, -1)

            # update the final positive and negative labels according to the user decision
            rej_ind_final[user_rej] = 1
            acc_ind_final[user_acc] = 1

            if get_feedback:
                out_image_file=dir+f'feedback_query_boxes_reviewed_gt_markup_step{step}.png'
                show_acc_rej_on_image(b_image,all_center_ind,acc_ind_final,rej_ind_final,out_image_file,convert_gray=False)

            
            if np.where(acc_ind_final == 1)[0].shape[0] != np.setdiff1d(np.where(acc_ind_final == 1)[0],np.where(rej_ind_final == 1)[0]).shape[0]:
                print("Should not get into thie function")
                import IPython; IPython.embed()

            sess.run(assign_rej_ind_final, {rej_box_ind_final_ph: rej_ind_final})
            sess.run(assign_acc_ind_final, {acc_box_ind_final_ph: acc_ind_final})
            sess.run(assign_rej_ind_smaller_final, {rej_box_ind_final_smaller_ph: calc_smaller_ind(np.where(rej_ind_final==1)[0],im_h,im_w,factor,True)})
            sess.run(assign_acc_ind_smaller_final, {acc_box_ind_final_smaller_ph: calc_smaller_ind(np.where(acc_ind_final==1)[0],im_h,im_w,factor,True)})

            user_rej = []
            user_acc = []
            cost_G = []
            """
            P: CLASSIFIER UPDATE
            """
            # Train the network using the updated labels
            print(f'Train the network using the updated labels')
            i=0
            while continue_training(rej_ind_final, acc_ind_final, curr_all_scores) and i<30:
                #print(f'updating models: iteration {i}')
                i += 1
                sess.run(opt_G_rejacc_maps)
                curr_all_scores = sess.run(all_scores)

            print(f'finished updating models after iteration {i}')
            continue_training(rej_ind_final, acc_ind_final, curr_all_scores,display=True)

            [curr_pos_ind, curr_ind_box] = sess.run([pos_ind, ind_box])
            curr_center_ind = all_center_ind[curr_ind_box]
            step += 1
        list_of_tuples = list(zip(Iteration, Ground_Truth_Count, Predicted_Count, Accuracy, precision, dens_sTILS, Time_Duration))
        list_of_tuples 
        df = pd.DataFrame(list_of_tuples,
                  columns = ['Iteration', 'Ground_Truth_Count','Predicted_Count', 'Recall (%)', 'Precision (%)', 'sTILS Density (%)','Time_Duration(sec)'])
        df.to_csv(dir+image_name+'_FINAL_Data.csv',index=False)
        """
        plt.plot(clicks_hist, Accuracy)
        plt.title('Feedback Corrections vs Recall')
        plt.xlabel('# Corrections')
        plt.ylabel('Recall (%)')
        plt.savefig(dir+image_name+'_Corrections_vs_Recall.png')
        """
        line_chartRecall = plt.plot(clicks_hist, Accuracy)
        line_chartPrec = plt.plot(clicks_hist, precision)
        plt.title('Feedback Corrections vs Recall & Precision')
        plt.xlabel('# Corrections')
        plt.ylabel('Recall & Precision(%)')
        plt.legend(['Recall', 'Precision'], loc=4)
        plt.savefig(dir+image_name+'_Corrections_vs_Recall_and_Precision.png')

    
def final(row_gt,col_gt,dir,step,input,ind_to_ask,acc_ind_final,init_time,clicks_counter,patch_sz,im_size):
    """
    The end of the algorithm: (1) save the input image with the repeating object on it. (2) save the repeating object in a .txt file.
    (3) calculate the localization error (4) print total time, clicks, false positive/negative, counting, ground_truth
    (5) save these measurments in a .txt file
    """
    print(f"================ eval step {step} ================")
    [count,final_res_image] = purple_dots_on_image(input,all_center_ind,ind_to_ask,acc_ind_final,patch_sz_hf)

    #image.imsave(dir+'final_res.png', scipy.misc.imresize(final_res_image,2.0))
    h,w=final_res_image.shape[0:2]
    #resized = cv2.resize(final_res_image,(2*w,2*h))
    print(f'saving fina result image to {dir}final_res{step}+.png, size wxh:{w}x{h},\
        from input size hxw:{input.shape[1]}x{input.shape[2]}')
    Iteration.append(step)
    image.imsave(dir+'final_res_%d.png'%(step), normalize_image(final_res_image))

    np.savetxt(dir + 'res_ours_' + str(step) + '.txt', np.concatenate((np.setdiff1d(ind_to_ask, np.where(acc_ind_final == 1)), np.where(acc_ind_final == 1)[0])))
    print("COUNT: ",count)
    stop_time = time.time()-init_time
    print("TOTAL TIME",stop_time)
    Time_Duration.append(stop_time)
    print("Number of clicks: ", clicks_counter)
    clicks_hist.append(clicks_counter)
    print(f"----------------------FP and FN step {step} ---------------")
    [FP,FN,gt,count_ours] = count_images(row_gt,col_gt,dir,step,patch_sz,im_size)

    print("gt_count: ", gt[0])
    print("our count: ", count_ours[0])
    Predicted_Count.append(count_ours[0])
    dens_sTILS.append(find_density(count_ours[0]))
    Ground_Truth_Count.append(gt[0])
    true_p = (gt[0] - FN[0])
    Accuracy.append(find_acc(gt[0], true_p))
    precision.append(find_prec(FP[0], true_p))
    print("FN: ", FN[0])
    print("FP: ", FP[0])

    print(str(np.round(stop_time,4)))

    output_file = open(dir + "/data_%d.txt"%(step), "a+")
    output_file.write("-----------------------------\n")
    output_file.write("Step: " + str(step) + "\n")
    output_file.write("TOTAL TIME: " + str(np.round(stop_time,4)) + "\n")
    output_file.write("Clicks : " + str(clicks_counter) + "\n")
    output_file.write("Count Ours : " + str(count_ours[0]) + "\n")
    output_file.write("gt : " + str(gt[0]) + "\n")
    output_file.write("FP : " + str(FP[0]) + "\n")
    output_file.write("FN : " + str(FN[0]) + "\n")
    output_file.write("-----------------------------\n")
    output_file.close()

    #plt.figure(10)
    #plt.imshow(normalize_image(final_res_image), cmap='gray');
    #plt.show()

if __name__ == "__main__":
    import sys
    global dataroot

    image_name = sys.argv[1]
    participant_name = sys.argv[2]
    print("image_name:",image_name)
    print("participant_name:",participant_name)
    
    #[im_size,window_loc,number_of_patches,th,path,show_gray,max_num_cells,gt_color,curr_patch_sz] = get_image_info(image_name)
    [im_size,number_of_patches,th,path,show_gray,max_num_cells,gt_color,curr_patch_sz,_] = get_image_info(image_name)
    patch_sz = curr_patch_sz
    #patch_sz_hf = np.int32(np.floor(curr_patch_sz / 2))
    patch_sz_hf = curr_patch_sz//2
    
    dataroot = '/content/drive/My Drive/Colab Notebooks/INBARHUB/'
    dir = dataroot+"user_study/"+str(participant_name)+"/"+str(image_name)+"/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(dir)
 #   main(dir,dir,im_size,image_name,window_loc,number_of_patches,th,path,gt_color,show_gray,max_num_cells,participant_name)
    main(dir,dir,im_size,image_name,number_of_patches,th,path,gt_color,show_gray,max_num_cells,participant_name)

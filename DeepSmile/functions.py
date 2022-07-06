#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy import linalg as LA

def ref_frame (path):
    df = pd.read_csv(path, usecols=range(0,324),header=None)
    
    # homogeneous transformation
    m1 = df[[0,1,2]].to_numpy()
    m2 = df[[3,4,5]].to_numpy()
    m3 = df[[6,7,8]].to_numpy()
    o = (m1[0]+m2[0]+m3[0])/3
    # compute the reference frame
    x_mag = LA.norm(m1[0]-o)
    x = (m1[0] - o)/x_mag
    y_mag = LA.norm(np.cross(m2[0] - m3[0], x))
    y = np.cross(m2[0] - m3[0], x) / y_mag
    z = np.cross(x, y)
    M = np.column_stack((x, y, z))
    M = [x, y, z] 
    Mo = np.column_stack((M, o))
    Mo_h = np.concatenate((Mo, [[0,0,0,1]])) #this is the homogeneous matrix
    
    # Certainly, there is a compact way to do this
    len_df_n = len(df) 
    len_df_col = len(df.columns)

    # "s" stands for step
    for i in range(len_df_n): 
        s_0 = df.loc[i] # take a row
        for j in range(3, int(len_df_col/3)):
            s_1 = s_0[3*j : 3*j+ 3] # select a 3D point at timestep j
            s_2 = s_1.to_numpy() # convert it from data frame to numpy array (vector)
            s_3 = np.append(s_2, 1) # convert it to homogeneous vector
            s_4 = np.reshape(s_3, (4,1)) # convert it as a column vector
            s_5 = np.matmul(Mo_h, s_4) # compute the transformation as homogeneous vector
            s_6 = np.squeeze(s_5) # transform it as a row vector
            s_7 = s_6[0:3] # select the first 3 elements since you don't need the homogeneous vector
            s_8 = pd.Series(s_7) # transform the vector to a Series in order to stack more Series
            if j==3:
                s_9 = s_8 # if it's the first element (3 row vector in the csv file), create it
            else:
                s_9 = s_9.append(s_8) # otherwise, append it to the first element of the corresponding row
            s_10 = s_9.to_frame() # transform it to a data frame
            s_11 = s_10.T # transpose it
        if i == 0:     
            s_12 = s_11 #if ithe vector does not exist, create it
        else:
            s_12 = s_12.append(s_11)
            
        data_ref = pd.DataFrame(s_12.to_numpy())
    return data_ref
        
def interpolate_signal(insignal, len1, len2):
    x1 = np.linspace(0, len2-1, len1)
    f = interp1d(x1, insignal, axis=0, kind='linear')
    x2 = np.linspace(0, len2-1, len2)
    return f(x2)
    
#inter_list = []
def list_to_interpolate(input_list, output_size=400):
    inter_list = []
    for elem in input_list:
        np_int = interpolate_signal(elem, elem.shape[0], output_size)
        inter_list.append(np.nan_to_num(np_int))
    return inter_list


def ref_to_dP0(data_ref):
    #df = data_ref.fillna(0)
    P = data_ref.to_numpy()
    num_it = len(P)
    num_cols_P = P[0].size
    num_cols_d = int(num_cols_P / 3.)
    d = np.zeros((num_it-1, num_cols_d))
    pa = np.zeros((0,3))
    for j in range(num_cols_d):
        for i in range(num_it-1):
            s = 3*j
            e = 3*j+3
            pa = P[i, s:e]
            p0 = P[0,s:e]
            d[i][j] = np.linalg.norm(pa - p0)
        
    return d

def list_to_array(input_list):
    np_list=[]
    for elem in input_list:
        df = pd.DataFrame(elem)
        df = df.iloc[:,[38, 40, 41,	42,	88,	89,	90,	91,	92,	83,	84,	85,	86,	87,	74,	75,	76,	77,	78,	79,	80,	81,	82,	72,	73,	67, # LEFT
                        98,	45,	44,	43,	46,	47,	48,	49,	50,	51,	52,	53,	54,	55,	56,	57,	58,	59,	60,	61,	62,	63,	64,	68,	69,	65]]

        np_array = df.to_numpy()
        np_list.append(np.nan_to_num(np_array))
    return np_list
    

def scaled_data2(input_list):
    max_value_list, scaled_list, min_value_list =list(),list(), list()
    
    for np_array in input_list:
        min_value = np.amin(np_array)
        new_array = np_array - min_value
        max_value = np.amax(new_array)
        output_array = new_array / max_value
        max_value_list.append(max_value)
        scaled_list.append(output_array)
        min_value_list.append(min_value)
    return scaled_list, max_value_list,min_value_list


def list_invers_scaled(scaled_list, max_value_list, min_value_list):
    invers_scaled_list= list()
    for i in  range(len(scaled_list)):
        output_array = (scaled_list[i]+ min_value_list[i] )* max_value_list[i]
        invers_scaled_list.append(output_array)
    return invers_scaled_list
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


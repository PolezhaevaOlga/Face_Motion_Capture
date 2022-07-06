# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:33:48 2022

@author: olgat
"""
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.data import Dataset
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import sklearn.utils._typedefs
import functions

def model_app (list_data, max_value_list, min_value_list):
    model_2 = load_model('lstm_3.h5')
    
    n_steps = 10
    
    def split_sequences(sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps # trouver la fin de sequense
            if end_ix > len(sequences)-1: # vérifier si on est au-delà du dataset
                break            
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :] # rassembler les parties d'entrée et de sortie
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)   
    
    
    
    feature_list =[]
    label_list = []
    for elem in list_data:
        feature, label = split_sequences(elem, n_steps)    
        feature_list.append(feature)
        label_list.append(label)
    feature_list  = np.array(feature_list)
    label_list = np.array(label_list)
    dataset_test = Dataset.from_tensor_slices((feature_list, label_list))
    
    
    rmse_test = list()
    for x, y in dataset_test:
        rmse = mean_squared_error(y, model_2.predict(x), squared=False)
        rmse_test.append(rmse)   
    rmse_test_mean = round(np.mean(rmse_test)*100,3)
        
    
    y_list, pred_list = list(), list()
    for x, Y in dataset_test:   
        y_list.append(Y.numpy())
        pred_list.append(model_2.predict(x))
    yhat = functions.list_invers_scaled(pred_list, max_value_list, min_value_list)
    y_real = functions.list_invers_scaled(y_list, max_value_list, min_value_list)
    rmse_test_mm = list()
    for i in range(len(y_real)):
        rmse_mm = mean_squared_error(y_real[i], yhat[i], squared=False)
        rmse_test_mm.append(rmse_mm )   
    rmse_mm = round(np.mean(rmse_test_mm),3)
    
    seuile =  0.05527
    
    return dataset_test, rmse_test_mean, rmse_mm 



# VISUALISATION

def multi_step_plot(dataset_test,rmse_test_mean,rmse_mm):
    
    model_2 = load_model('lstm_3.h5')
 
    for x, y in dataset_test:
        true_future = y
        prediction = model_2.predict(x)

        
        fig = plt.figure(figsize=(9, 10))
    #     fig = plt.figure(figsize=(10, 15))
        num_out = len(true_future)
        
        plt.subplots_adjust(wspace= 0.3, hspace=0.5)
        plt.subplot(4, 2, 1)
        
    
        plt.plot(np.arange(num_out), np.array(true_future[:, 1]), color='royalblue', lw=2, label='real')
        plt.plot(np.arange(num_out), np.array(prediction[:, 1]), 'darkorange',lw=2,label='predicted',)
        plt.fill_between(np.arange(num_out), np.array(true_future[:, 1]).reshape(num_out), np.array(prediction[:, 1]).reshape(num_out), 
                         color='wheat',label='error')
        #plt.ylim(0, 25)
        plt.ylim(0,1.1)
        plt.title('Left Commissure', fontname="Times New Roman",fontsize=15)
        plt.ylabel('S(t)', fontname="Times New Roman",fontsize=15)
        plt.xlabel('Time [sec]', fontname="Times New Roman",fontsize=15) 
        plt.legend(labels=["Input", "Prediction", "Error"],fontsize=10, loc='lower center') 
        plt.tick_params(axis='both', which='major', labelsize=10)
    
        plt.subplot(4, 2,2)
     
        plt.plot(np.arange(num_out), np.array(true_future[:, 27]), color='royalblue', lw=2,
                   label='real ')
       
        plt.plot(np.arange(num_out), np.array(prediction[:, 27]), 'darkorange',lw=2,
                         label='predicted ')
        plt.fill_between(np.arange(num_out), np.array(true_future[:, 27]).reshape(num_out), 
                         np.array(prediction[:, 27]).reshape(num_out), color='wheat',label='erreur')
        plt.ylim(0, 1.1)
        plt.title('Right Commissure', fontname="Times New Roman",fontsize=15)
        plt.ylabel('S(t)', fontname="Times New Roman",fontsize=15)
        plt.xlabel('Time [sec]', fontname="Times New Roman",fontsize=15) 
        plt.tick_params(axis='both', which='major', labelsize=10)
    #     plt.suptitle('Upper mucocutaneous line')
        
        plt.subplot(4, 2, 3)
        
        plt.plot(np.arange(num_out), np.array(true_future[:, 2]), color='royalblue', lw=2, label='real')
        plt.plot(np.arange(num_out), np.array(prediction[:, 2]), 'darkorange',lw=2,
                     label='predicted',)
        plt.fill_between(np.arange(num_out), np.array(true_future[:, 2]).reshape(num_out), np.array(prediction[:, 2]).reshape(num_out), 
                         color='wheat',label='error')
        #plt.ylim(0, 25)
        plt.ylim(0,1.1)
        plt.title('Left Upper mucocutaneous line', fontname="Times New Roman",fontsize=15)
        plt.ylabel('S(t)', fontname="Times New Roman",fontsize=15)
        plt.xlabel('Time [sec]', fontname="Times New Roman",fontsize=15) 
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.subplot(4, 2, 4)
        plt.plot(np.arange(num_out), np.array(true_future[:, 28]), color='royalblue', lw=2,label='real')
        plt.plot(np.arange(num_out), np.array(prediction[:, 28]), 'darkorange',lw=2,label='predicted ')
        plt.fill_between(np.arange(num_out), np.array(true_future[:, 28]).reshape(num_out), 
                         np.array(prediction[:, 28]).reshape(num_out), color='wheat',label='erreur')
        plt.ylim(0, 1.1)
        plt.ylabel('S(t)', fontname="Times New Roman",fontsize=15)
        plt.xlabel('Time [sec]', fontname="Times New Roman",fontsize=15)  
        plt.title(' Right Upper mucocutaneous line ', fontname="Times New Roman",fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
       
        plt.subplot(4, 2, 5)
        plt.plot(np.arange(num_out), np.array(true_future[:, 40]), color='royalblue', lw=2, label='real')
        plt.plot(np.arange(num_out), np.array(prediction[:, 40]), 'darkorange',lw=2,label='predicted',)
        plt.fill_between(np.arange(num_out), np.array(true_future[:, 40]).reshape(num_out), np.array(prediction[:, 40]).reshape(num_out), 
                         color='wheat',label='error')
    
        plt.ylim(0,1.1)
        plt.title('Left Triangular lip line', fontname="Times New Roman",fontsize=15)
        plt.ylabel('S(t)', fontname="Times New Roman",fontsize=15)
        plt.xlabel('Time [sec]', fontname="Times New Roman",fontsize=15) 
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.subplot(4, 2, 6)
        plt.plot(np.arange(num_out), np.array(true_future[:, 14]), color='royalblue', lw=2,label='real ')
        plt.plot(np.arange(num_out), np.array(prediction[:, 14]), 'darkorange',lw=2, label='predicted ')
        plt.fill_between(np.arange(num_out), np.array(true_future[:, 14]).reshape(num_out), 
                         np.array(prediction[:, 14]).reshape(num_out), color='wheat',label='erreur')
        
        plt.ylim(0, 1.1)
        plt.title('Right Triangular lip line', fontname="Times New Roman",fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('S(t)', fontname="Times New Roman",fontsize=15)
        plt.xlabel('Time [sec]', fontname="Times New Roman",fontsize=15)    
        
        # plt.subplot(4, 1, 1)
        # plt.title('RESULT', fontname="Times New Roman",fontsize=15)
        # plt.axis([0, 6, 0, 6])
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # im = plt.imread('logo.png')
        # t = ( f'Surname: {sujet_name} \n Degree of abnormality: {rmse_test_mean} % \n Degree of abnormality in millimeters: {rmse_mm} mm'
        #  )
        # plt.text(3, 3, t, fontsize=13, style='normal', horizontalalignment='center',
        #     verticalalignment= 'center',)
        # newax = fig.add_axes([0.7,0.7,0.21,0.21], anchor='NE',zorder=1)
        # newax.imshow(im)
        # newax.axis('off')

        # plt.box('on')
        # plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
     
        
        return fig,plt
    
        

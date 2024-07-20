# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 23:17:09 2024

@author: anujb

These contain helper functions for VIN UNI tissue image processing 
"""

#HELPER FUNCTIONS TO LOAD THE RAW DAT FILES AND CONVERT THEM TO I AND Q matrices
 
def loadSavedRawData(file_name):
    f = open(file_name, 'rb')
    MYDAT = f.read()
    f.close()
    I_RAW, Q_RAW = convertToIQImage(MYDAT)
    I_ADC, Q_ADC, I_VOLTS, Q_VOLTS = convertADCToVolts(I_RAW, Q_RAW)
    return I_ADC, Q_ADC, I_VOLTS, Q_VOLTS


#take raw byte_data and convert to ADC output (bit shift is NOT corrected)
def convertToIQImage(byte_data):
    import numpy as np
    wi = 0
    imgBytesI = np.zeros(128*128)
    imgBytesQ = np.zeros(128*128)
    for row in range (128):
        for col in range(128):
            wi = row*128 + col
            iwrd = (byte_data[4 * wi + 0] + 256*byte_data[4 * wi + 1])
            qwrd = (byte_data[4 * wi + 2] + 256*byte_data[4 * wi + 3])
            imgBytesI[wi] = iwrd
            imgBytesQ[wi] = qwrd
            
    J_MYIMAGE_I=imgBytesI.reshape([128,128])
    J_MYIMAGE_Q=imgBytesQ.reshape([128,128])
    return J_MYIMAGE_I, J_MYIMAGE_Q


def convertADCToVolts(I_IMAGE, Q_IMAGE):
    I_IMAGE_ADC = I_IMAGE/16 #correct bit shift
    Q_IMAGE_ADC = Q_IMAGE/16 #correct bit shift
    I_IMAGE_VOLTS = I_IMAGE_ADC*1e-3 #convert to volts
    Q_IMAGE_VOLTS = Q_IMAGE_ADC*1e-3 #convert to volts
    return I_IMAGE_ADC, Q_IMAGE_ADC, I_IMAGE_VOLTS, Q_IMAGE_VOLTS

















#%%MORE HELPER FUNCTIONS FOR POST PROCESSING

def impedance_si(ref_coef, array = True):
    
    csi = 8433
    psi = 2329
    zsi = csi*psi
    if array == False:
        zsamp = (zsi*(1-ref_coef))/(1+ref_coef)
        return zsamp/1e6
    if array ==True:
        new_list = []
        for jj in ref_coef:
            zsamp = (zsi*(1-jj))/(1+jj)
            new_list.append(zsamp/1e6)
        return new_list
    
    
#%%helper functions for plotting

import matplotlib.pyplot as plt

def multi_image_ROWCOL(frames, titles, rows, cols, vmin=None, vmax=None, cmap=None):
    """
    Display multiple frames in a grid format with optional parameters.

    Parameters:
    - frames: List of image frames (arrays).
    - titles: List of titles for the images.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    - vmin: Minimum data value for colormap scaling.
    - vmax: Maximum data value for colormap scaling.
    - cmap: Colormap to use for displaying the images.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.ravel()  # Flatten the 2D array of axes to 1D for easy iteration

    for i in range(rows * cols):
        if i < len(frames):
            ax = axes[i]
            ax.imshow(frames[i], vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(titles[i])
            ax.axis('off')
        else:
            axes[i].axis('off')  # Hide axes if there are more subplots than images

    plt.tight_layout()
    plt.show()
    
def multi_image(mat_list, str_list, vmin = -0.1, vmax = 0.1, cmap = "Spectral"):
    num1 = len(mat_list)
    if num1 % 2 == 0:
        if num1 != 2:
            row = num1//2
            col = num1//2
        else:
            row = num1//2
            col = num1//2 + 1
    else:
        row = (num1//2) + 1
        col = (num1//2) + 1

    # create figure
    fig = plt.figure()

    for jj in range(num1):
        fig.add_subplot(row, col, jj+1)
        plt.imshow(mat_list[jj], interpolation='bilinear', vmin = vmin, vmax = vmax, cmap = cmap)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.title(str_list[jj])
        plt.colorbar()
        
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:36:56 2024

@author: s164633
"""

def mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os

    if not os.path.exists(directory):
        os.makedirs(directory)

    return []

import os
from scipy.io import loadmat
import numpy as np 
import skimage.io as skio
import pylab as plt
import skimage.transform as sktform
import skimage.segmentation as sksegmentation
import skimage.morphology as skmorph
import pandas as pd

# Condition and experiments
condition = 'glass'
exp = '240530_exp3'

# pixel resolution
px = 0.0602000

# Determine corresponding rawRoot 
rawRoot = os.path.join('/project/bioinformatics/Danuser_lab/3Dmorphogenesis/raw/tisogai/TIRF-2',exp+'_U2OS_VCLARPC2-PLA',condition)

# Determine analysisRoot and its folders

analysisRoot = os.path.join('/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/tisogai/3D-ARP3 Project/PLA_VCL-ARPC2',exp,'analysis',condition)
Afolders = sorted([f for f in os.listdir(analysisRoot) if os.path.isdir(os.path.join(analysisRoot, f))])
for folder in np.arange(len(Afolders))[:]:
    maskfolders = os.path.join(analysisRoot, Afolders[folder], 'cyto3')
    PLAfolders = os.path.join(analysisRoot, Afolders[folder], 'DotDetection')
    
    
    # load the masks generated using cellpose 
    imgname = Afolders[folder]
    
    maskfile = os.path.join(maskfolders, 
                            imgname+'_segmentation.tif')

    masks = skio.imread(maskfile)
    
    # load the original vinculin images to later determine whether its EV or Tagged-VCL constructs
    basename_of_raw = imgname.split('_')[0]
    rawfile = os.path.join(rawRoot, basename_of_raw, imgname+'_w1488-WF.TIF')

    raw = skio.imread(rawfile)
    
    # load the x,y coordinates from the PLA detection process
    try: 
        xy_coord = loadmat(os.path.join(PLAfolders,
                       imgname+'.mat'))
    except: 
        xy_coord = {}
        xy_coord['x'] = np.zeros((raw.shape),dtype= 'float64')
        xy_coord['y'] = np.zeros((raw.shape),dtype= 'float64')
    
    
    # load PLA image to retrieve the image dimension    
    plafile = os.path.join(rawRoot, basename_of_raw, imgname+'_w3642-WF.TIF')
    pla_img = skio.imread(plafile)
    
    # resize mask to PLA image size dimensions, and remove small masks and masks touching the image borders
    resized_masks = sktform.resize(masks, 
                                    output_shape=pla_img.shape, 
                                    preserve_range=True, 
                                    order=0) # nearest neighbor interpolation   
    resized_masks = skmorph.remove_small_objects(resized_masks, min_size=100)
    resized_masks = sksegmentation.clear_border(resized_masks)
    
    # initialize variables
    mask_ids =[]
    centroids = []
    cell_areas = []
    PLA_in_a_mask = []
    PLA_densities = []
    
    # Iterate over unique cell ids in the masks and determine masked area and PLA in masked area
    cell_ids = np.setdiff1d(np.unique(resized_masks), 0)
    
    for cell_id in cell_ids:
        
        # record the id of the mask and centroid of masks
        mask_id = cell_id
        mask_ids.append(mask_id)
        
        select_mask = resized_masks==cell_id
        coord_mask = np.argwhere(select_mask>0)
        centroid = np.mean(coord_mask,0)
        centroids.append(centroid)

        
        # determine cell area per mask in um2
        area = np.sum(select_mask)*pow(px,2)
        cell_areas.append(area)
        
        # determine number of PLAs per mask
        dots_in_a_mask = select_mask[xy_coord['y'].astype(np.int64), xy_coord['x'].astype(np.int64)]
        PLA_in_a_mask.append(np.sum(dots_in_a_mask))
        
        # determine the PLA density (#/um2) per cellmask
        PLA_density = np.sum(dots_in_a_mask)/area
        PLA_densities.append(PLA_density)
    
    """
    Restructure the data and save as a table
    """
    saveFolder = os.path.join(analysisRoot, Afolders[folder], 'PLA-quantification')
    mkdir(saveFolder) 
    
    mask_ids_tableCSV = np.array(mask_ids)
    cell_area_tableCSV = np.array(cell_areas)
    PLA_per_cellsCSV = np.array(PLA_in_a_mask)
    PLA_densitiesCSV = np.array(PLA_densities)
    
    df = pd.DataFrame({'Mask ID': mask_ids_tableCSV,
                       'Cell Area': cell_area_tableCSV, 
                      'PLA per cell': PLA_per_cellsCSV, 
                      'PLA density per cell': PLA_densitiesCSV}, 
                      index=None)
    
    df.to_csv(os.path.join(saveFolder,imgname+'_PLAquant.csv'), index=False)

        
    """
    Save the mask boundaries overlayed on the vinculin image, with detected PLA dots within masks in red
    """
    
    mask_overlay = sksegmentation.mark_boundaries(np.dstack([raw,
                                                             raw,
                                                             raw])/raw.max(),
                                            resized_masks, mode={'outer', 'thick'})
    
    # Accumulate all PLA dots found in all masks
    dots_in_all_masks = resized_masks[xy_coord['y'].astype(np.int64), xy_coord['x'].astype(np.int64)].astype(bool)
    
    all_dots_in_masks = {}
    all_dots_in_masks['x'] = xy_coord['x'][dots_in_all_masks]
    all_dots_in_masks['y'] = xy_coord['y'][dots_in_all_masks]

    # save figures with mask boundaries and all PLA dots within
    plt.imshow(mask_overlay)
    plt.plot(all_dots_in_masks['x'], all_dots_in_masks['y'], 'ro', markersize=0.35)
    plt.savefig(os.path.join(saveFolder,imgname+'_PLA_and_mask_overlay.eps'), format = 'eps')
    plt.show()
    
    # save figures with mask boundaries and mask labels     
    plt.imshow(mask_overlay)
    for i in range(len(mask_ids)):
        k = centroids[i]
        plt.text(k[1], k[0], mask_ids[i], fontsize='xx-large', c='c')

    mask_folder = os.path.join('/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/tisogai/3D-ARP3 Project/PLA_VCL-ARPC2', 'mask_folder',exp,condition)
    mkdir(mask_folder) 
    plt.savefig(os.path.join(mask_folder,imgname+'_Mask_ID_overlay.png'))
    plt.show()
    
   
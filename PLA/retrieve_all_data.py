#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:26:52 2024

@author: s164633
"""

"""
Retrieve all quantifications and assemble a 
master csv file
"""
import os 
import numpy as np
import pandas as pd

# Condition and experiments
condition = 'soft'
exp = '240520_exp2'

# Determine the location of csv files with analyzed data
analysisRoot = os.path.join('/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/tisogai/3D-ARP3 Project/PLA_VCL-ARPC2',exp,'analysis',condition)
Afolders = sorted([f for f in os.listdir(analysisRoot) if os.path.isdir(os.path.join(analysisRoot, f))])


# fix savePath
saveFolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/tisogai/3D-ARP3 Project/PLA_VCL-ARPC2'

# loop through all csv files and construct final csv file
for folder in np.arange(len(Afolders))[:]:
    quantFolder = os.path.join(analysisRoot, Afolders[folder], 'PLA-quantification')
      
    # load the masks generated using cellpose 
    dataname = Afolders[folder]
    
    filename = [file for file in os.listdir(quantFolder) if file.endswith(".csv")]
 
    datafile = os.path.join(quantFolder,filename[0])
    
    df = pd.read_csv(datafile,encoding='utf-8-sig')
    df.insert(0,"Image File",dataname,True)
    
    df.to_csv(os.path.join(saveFolder, exp+ '_'+ condition+ '.csv'), mode='a', header=False, index=False, encoding='utf-8-sig')

df_final = pd.read_csv(os.path.join(saveFolder, exp+ '_'+ condition+ '.csv'), encoding='utf-8-sig')
column_names = ['Image File', 'Mask ID', 'Cell Area', 'PLA per cell', 'PLA density per cell']
df_final.columns = column_names


df_final.to_csv(os.path.join(saveFolder, exp+ '_'+ condition+ '.csv'), mode='w', index=False, encoding='utf-8-sig')

    
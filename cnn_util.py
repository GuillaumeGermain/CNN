#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:42:19 2018

@author: guillaume
"""

import h5py

def print_file_layers(filename):
    """
    show layers stored in a h5 file
    
    filename: h5 file to open
    """
    #Print layers in the file

    with h5py.File(filename) as f:
        #print(list(f))
        for i in list(f):
            print(i)
    


filename = "classifier_final.h5"
print_file_layers(filename)

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:26:39 2018

@author: KAT7RNG
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *

import iSVD_module as mod
import SU2_iSVD_module as SU2mod

#----------------------------------------------------------------------#
# main()
k = 15 # number of initial snapshots
#----------------------------------------------------------------------#
base_string = 'restart_flow'
file_extension = '.dat'

#base_string = 'restart'
#file_extension = '.dat'

files = SU2mod.create_filename_array(base_string,file_extension)

base_string = 're1_'+str(k)
out_files1 = SU2mod.create_filename_array(base_string,file_extension)

base_string = 're2_'+str(k)
out_files2 = SU2mod.create_filename_array(base_string,file_extension)

#files = ['solution_flow_00000.dat','solution_flow_00001.dat']
#out_files = ['re_solution_flow_00000.dat','re_solution_flow_00001.dat']

print(files[0], files[9], out_files1[0], out_files1[9])
#----------------------------------------------------------------------#
primaries = np.array([14,17,20,23,26,29,32])
primaries = ['Conservative_1','Conservative_2','Conservative_3','Pressure']
file_counter = 0

for line_number in primaries:
    for i in range(0,k):
        #v = SU2mod.extract_vector(files[i],line_number)
        v = SU2mod.extract_vector2(files[i],line_number)
        print(type(v),v.shape)
        if i == 0: A = v
        else: A = np.c_[A,v]
        print(i)
    
    print(A.shape, A[A.shape[0]-1][0])
    U, s, V = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    U, S ,V = mod.reshape_SVD(U,S,V,k)
    
    print(np.allclose(A, U@S@V))

    # loop over new snapshots
    #for i in range(k,len(files)):
    for i in range(k,98):
        # after a new primal the solution w has to be added
        #v = SU2mod.extract_vector(files[i],line_number)
        v = SU2mod.extract_vector2(files[i],line_number)
        A = np.c_[A,v]
        
        U, S, V = mod.add_vector_to_SVD(U, S, V, v)
        U, S, V = mod.reshape_SVD(U, S , V, k)
        print('iSVD performed on:',i)

    # Reconstruction of the full dataset, write output
    A_re_iSVD = U@S@V
    #for i in range(0,len(files)):
    for i in range(0,98):
        v = A_re_iSVD[:,[i]]
        print('Write file number:',i)
        if file_counter == 0:
            #SU2mod.write_reconstructed_result(v,files[i],out_files1[i],line_number)
            SU2mod.write_reconstructed_result2(v,files[i],out_files1[i],line_number)
        elif file_counter % 2 == 1:
            SU2mod.write_reconstructed_result2(v,out_files1[i],out_files2[i],line_number)
            #SU2mod.write_reconstructed_result2(k,v,out_files1[i],out_files2[i],line_number)
        else:
            SU2mod.write_reconstructed_result2(v,out_files2[i],out_files1[i],line_number)
            #SU2mod.write_reconstructed_result2(k,v,out_files2[i],out_files1[i],line_number)
            
    file_counter += 1

    #U,s,V=np.linalg.svd(A,full_matrices=False)
    #S = np.diag(s)
    #U, S, V = mod.reshape_SVD(U, S , V, k)
    #A_re_SVD = U@S@V
    #Diff = np.absolute(A - U@S@V)


    #print('iSVD=',mod.rms_error(A,A_re_iSVD))
    #print('SVD=',mod.rms_error(A,A_re_SVD))
    #print('Diff iSVD SVD=',np.mean(np.absolute(A_re_iSVD-A_re_SVD)))
    #print('RMS Error iSVD SVD=',mod.rms_error(A_re_iSVD,A_re_SVD))

    #plot_2D_Data(A,A_re_SVD,A_re_iSVD)

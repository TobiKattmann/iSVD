# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:27:33 2018

@author: KAT7RNG
"""
import numpy as np
import math

def add_vector_to_SVD(U,S,V,v):
    """U@S@V: current SVD, and v is vector to be included vi iSVD.  
    
    Incremental SVD after Balzano2013 (theory) (or Vezyris2014 (application)).
    In Args, n: number DOF's, k: number of modes, m: current number of timesteps.

    Args: 
      U (np.ndarray(n,k)): left matrix
      S (np.ndarray(k,k)): singular value matrix
      V (np.ndarray(k,m)): right matrix
      v (np.ndarray(n,)): vector to be added to the SVD
  
    Returns:
      (np.ndarray(n,k)): left matrix
      (np.ndarray(k,k)): singular value matrix
      (np.ndarray(k,m)): right matrix
    """
    if (v.ndim != 1 or
        not np.allclose(S,S.T) 
        ): raise Exception('Dimension Error')
    w = U.T@v
    p = U@w
    r = v - p
    
    update_matrix = np.c_[S,w]
    update_matrix = np.r_[update_matrix, np.zeros((1,update_matrix.shape[1]))]
    update_matrix[update_matrix.shape[0]-1][update_matrix.shape[1]-1] = np.linalg.norm(r)
    
    U_hat, s_hat, V_hat = np.linalg.svd(update_matrix, full_matrices=False)
    S_hat = np.diag(s_hat)
    
    U = (np.c_[U,r/np.linalg.norm(r)])@U_hat
    S = S_hat
    
    V_tmp = np.c_[V.T,np.zeros(V.shape[1])]
    V_tmp = np.r_[V_tmp,np.zeros((1,V_tmp.shape[1]))]
    V_tmp[V_tmp.shape[0]-1][V_tmp.shape[1]-1] = 1
    V = V_tmp@V_hat.T
    
    return (U,S,V.T)

def reshape_SVD(U,S,V,k):
    """SVD-matrices are reshaped such that only the first k moddes are kept.
  
    Args: 
      U (np.ndarray(n,k)): left matrix
      S (np.ndarray(k,k)): singular value matrix
      V (np.ndarray(k,m)): right matrix
      k (int): cut off dimension

    Returns:
      (np.ndarray(n,k)): left matrix
      (np.ndarray(k,k)): singular value matrix
      (np.ndarray(k,m)): right matrix
    """
    while S.shape[0] > k: 
        U = np.delete(U,U.shape[1]-1,1)
        
        S = np.delete(S,S.shape[1]-1,1)
        S = np.delete(S,S.shape[0]-1,0)
        #s = np.diag(S) # Diagonal matrix becomes vector      

        V = np.delete(V,V.shape[0]-1,0)
    return (U, np.diag(S), V)

def rms_error(A,B):
    """Compute roots mean square error between two matrices.

    Args:
      A (np.ndarray): matrix
      B (np.ndarray): matrix
    Returns:
      float: root mean square error
    """
    C = A-B
    C = C*C
    sum = np.sum(C)
    rms = math.sqrt(sum / (A.shape[0]*A.shape[1]))
    return rms
    

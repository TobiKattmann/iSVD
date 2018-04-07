import numpy as np
import pandas as pd
import copy

import iSVD_module as iSVD_func

"""
Best practice
-call this python script with all restart_flow_*.dat files in the dir
-for visualization with tecplot convert ASCII to .plt via
  find . -name "re_*.dat" -exec preplot {} \;
"""

#########################################################################
class IO:
  """Class which handles IO with csv-files."""

  def extract_vectors(self, filename, var_names):
    """Extracts np.arrays from file (csv-format).
    
    Args:
      filename (str): name of the file to be read
      var_names (list of str): contains the names of variables to be parsed
    Returns:
      list of np.ndarray (n,): each list entry contains a vectors with parsed numbers 
    """
    df = pd.read_csv(filename, delim_whitespace=True) # read whole csv into dataframe
    V = df.as_matrix(columns=[var_names])

    return_list = []
    for i in range(len(var_names)):
      return_list.append(V[0:V.shape[0]-5,i]) # 5 because theres 5 additional lines in the file which is not data

    return return_list # is list with a nd.array of shape (number,) in each item

  #----------------------------------------------------------------------# 
  def rewrite_file(self, in_filename, out_filename, var_names, in_vectors):
    """Copy original file and rewrite new values in respective columns.

    Args:
      in_filename (str): old file which is partly copied
      out_filename (str): name of the new file
      var_names (list of str): contains the names of variables to be replaced
      in_vectors (list of np.ndarray (n,)): contains replacing data
    """
    if len(var_names)!=len(in_vectors): raise Exception('Dimension mismatch.')
    # read original file
    df = pd.read_csv(in_filename, delim_whitespace=True)
  
    # replace data vector
    for i in range(len(var_names)):
      new_df = pd.DataFrame({var_names[i]:in_vectors[i]})
      df.update(new_df)

    # add quotes to header variables as they are normally dismissed
    var_names = df.columns.values.tolist()
    for i in range(len(var_names)):
        var_names[i] = '"'+var_names[i]+'"' 
    df.columns = var_names
    # write new csv-file
    df.to_csv(out_filename, sep=' ', index=False, quotechar=' ')
    
#########################################################################

class iSVD:
  """Performs incremental SVD."""
  def __init__(self, var_names, num_modes=15, base_string='restart_flow', file_extension='.dat'):
    self.k = num_modes # number of base vectors
    self.base_string = base_string # of input file
    self.file_extension = file_extension # of input file
    self.var_names = var_names # variable names to be compressed
    
    self.num_vars = len(self.var_names)
    self.in_files = self.create_filename_array(self.base_string, self.file_extension)
    self.num_files = len(self.in_files) 
    self.num_files = 98
    self.base_string_out = 're'
    self.out_files = self.create_filename_array(self.base_string_out, self.file_extension)
    if len(self.in_files)!=len(self.out_files): raise Exception("There must be as many in as out files")
    self.decomp = [[None]*3 for i in range(self.num_vars) ] # contains SVD decomposition for all variables; 3=U,S,V;

  #----------------------------------------------------------------------# 
  def create_filename_array(self, base_string, file_extension):
    """Create array which contains all input/output file names.

    Args:
      base_string (str): first part of filename
      file_extension (str): filename ending (e.g. .dat)
    Returns:
      list of str: contains all filenames
    """
    # 00002-02499
    counter = 0 # for array index 

    files = []
    for i in range(2,10):
        filename = base_string+'_0000'+str(i)+file_extension
        files.append(filename)
        counter += 1

    for i in range(10,100):
        filename = base_string+'_000'+str(i)+file_extension
        files.append(filename)
        counter += 1

    for i in range(100,1000):
        filename = base_string+'_00'+str(i)+file_extension
        files.append(filename)
        counter += 1

    for i in range(1000,2500):
        filename = base_string+'_0'+str(i)+file_extension
        files.append(filename)
        counter += 1
      
    return files

  #----------------------------------------------------------------------#
  def perform_iSVD(self):
    """Creates iSVD over time.

    Initially creates an exact SVD with the first k sanpshots.
    Then each new snapshot is directly incoorporated via iSVD.
    """
    self.initial_SVD()
    io = IO()

    for i in range(self.k, self.num_files): 
      print("iSVD for file: ", i)
      # extract all vectors of file at once: multiple np.arrays
      V = io.extract_vectors(self.in_files[i], self.var_names)
      for j in range(self.num_vars):
        # add vector to SVD in self.decomp
        U_tmp, S_tmp, V_tmp = iSVD_func.add_vector_to_SVD(self.decomp[j][0], np.diag(self.decomp[j][1]) , self.decomp[j][2], V[j])
        # reshape_SVDin self.decomp
        self.decomp[j][0], self.decomp[j][1] , self.decomp[j][2] = iSVD_func.reshape_SVD(U_tmp, S_tmp, V_tmp, self.k)
    # Done
  #----------------------------------------------------------------------#

  def initial_SVD(self):
    """Creates initial exact SVD on first k snapshots."""
    io = IO()
    A = [None]*self.num_vars # A_full = [A,A,...] after filling 
    for i in range(self.k):
      # extract all vectors of a file at once. 
      V = io.extract_vectors(self.in_files[i], self.var_names)
      # add them to the appropriate entry of A_full
      for j in range(self.num_vars):
        if i==0:
          A[j] = V[j]
        else:
          A[j] = np.c_[A[j],V[j]]

    for i in range(self.num_vars):
      # perform SVD on each entry of A_full and store result in self.decomp
      U,s,V = np.linalg.svd(A[i], full_matrices=False)

      #self.decomp[i][0] = copy.deepcopy(U)
      #self.decomp[i][1] = copy.deepcopy(s)
      #self.decomp[i][2] = copy.deepcopy(V)
    
      self.decomp[i][0] = U
      self.decomp[i][1] = s
      self.decomp[i][2] = V
    #test
    for i in range(self.num_vars):
      print(np.allclose(A[i], self.decomp[i][0]@np.diag(self.decomp[i][1])@self.decomp[i][2] ))

  #----------------------------------------------------------------------#
  def reconstruct_data_from_iSVD(self):
    """Writes reconstruted results back into restart-files.

    Full dataset is restored and stored and written into new files
    """
    io = IO()
    # Reconstruct full data
    full_data = [None]*self.num_vars
    for i in range(self.num_vars):
      full_data[i] = self.decomp[i][0]@np.diag(self.decomp[i][1])@self.decomp[i][2]

    for i in range(self.num_files):
      print("Write file: ", i)
      # create reconstructed results 'numvars * np.array'
      V = [full_data[j][:,i] for j in range(self.num_vars)]
      # write these vectors back in file
      io.rewrite_file(self.in_files[i], self.out_files[i], self.var_names, V)

#########################################################################
if __name__ == "__main__":
  primaries = ['Conservative_1','Conservative_2','Conservative_3','Pressure']

  MOR = iSVD(num_modes=15, var_names=primaries)
  MOR.perform_iSVD()
  MOR.reconstruct_data_from_iSVD()


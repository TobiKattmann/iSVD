import numpy as np
import pandas as pd
#----------------------------------------------------------------------#
def create_filename_array(base_string,file_extension):
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
# Follwing two functions work on the ascii vtk output
def extract_vector(filename,line_number):
    with open(filename,'r') as f:
        lines = f.readlines()
    string_array = lines[line_number].split()
    v = np.array(list(string_array), dtype=float)
    return v
#----------------------------------------------------------------------#    
def write_reconstructed_result(k,v,in_filename,out_filename,line_number):
    v = np.squeeze(np.asarray(v)) # convert (.,1) to (.,) array
    counter = 0 # to identify the correct line while copying the file
    
    with open(in_filename,'r') as original:
        with open(out_filename,"w") as output: 
            for line in original:
                if counter != line_number:
                    output.write(line)
                else:
                    for i in range(0,len(v)):
                        output.write(str(v[i])+" ")
                    output.write("\n")
                counter += 1
    return

#----------------------------------------------------------------------#
# Following two functions work on the restart-files in order to feed
# them back in the adjoint solver

# TODO line number will nork work as the information is stored in a 
# coloum-vector
def extract_vector2(filename,var):
    df = pd.read_csv(filename,delim_whitespace=True)
    v = df.as_matrix(columns=[var])
    v = np.squeeze(np.asarray(v))
    return v[0:len(v)-5] # 5 because theres 5 additional lines in the file which is not data
#----------------------------------------------------------------------# 
# TODO line number will nork work as the information is stored in a 
# coloum-vector   
def write_reconstructed_result2(v,in_filename,out_filename,var):
    v = np.squeeze(np.asarray(v)) # convert (.,1) to (.,) array
    # replace data vector
    df = pd.read_csv(in_filename,delim_whitespace=True)
    new_df = pd.DataFrame({var:v})
    df.update(new_df)
    # add quotes to header variables as they are normally dismissed
    var_names = df.columns.values.tolist()
    for i in range(0,len(var_names)):
        var_names[i] = '"'+var_names[i]+'"' 
    df.columns = var_names
    df.to_csv(out_filename, sep=' ', index=False, quotechar=' ')
    return
#----------------------------------------------------------------------# 

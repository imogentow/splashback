import numpy as np

def convert(filename):
    halo_IDs = np.genfromtxt("../Analysis_files/halo_IDs.csv", delimiter=",")
    
    to_convert = np.genfromtxt(filename, delimiter=",")
    
    N_clusters = len(to_convert)
    converted = np.zeros(N_clusters)
    for i in range(N_clusters):
        converted[i] = np.where(halo_IDs == to_convert[i])[0]
        
    return converted
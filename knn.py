import numpy as np
from scipy.spatial import distance

def distance_au_centre(X):
    center = 0.5 * np.ones(len(X[0]))
    array = distance.cdist(X, center, 'chebyshev')



    
def voisin_plus_proche(X):

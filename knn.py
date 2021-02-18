import numpy as np
from scipy.spatial import distance

def array_center(X):
    center = [0] * 0
    center.append(0.5 * np.ones(len(X[0])))
    return distance.cdist(X, center, 'chebyshev')

def distance_au_centre(X):
    array = array_center(X)
    somme = 0
    for ave in array:
        somme += ave
    return somme / len(array)

    
def voisin_plus_proche(X):
    array = array_center(X)
    return np.amin(array)


for d in range(1,21):
          dist = []
          v = []
          for i in range(10):
              X = np.random.rand(100,d)
              dist.append(distance_au_centre(X))
              v.append(voisin_plus_proche(X))
          print(np.mean(dist), np.mean(v))

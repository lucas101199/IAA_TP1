import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

"""
def array_center(X) :
    center = [0] * 0
    center.append(0.5 * np.ones(len(X[0])))
    return distance.cdist(X, center, 'chebyshev')


def distance_au_centre(X):
    array = array_center(X)
    somme = 0
    for ave in array :
        somme += ave
    return somme / len(array)


def voisin_plus_proche(X):
    array = array_center(X)
    return np.amin(array)


for d in range(1, 21):
    dist = []
    v = []
    for i in range(10) :
        X = np.random.rand(100, d)
        dist.append(distance_au_centre(X))
        v.append(voisin_plus_proche(X))
    print(np.mean(dist), np.mean(v))
"""

def damier(dimension, grid_size, nb_examples, noise=0) :
    data = np.random.rand(nb_examples, dimension)
    labels = np.ones(nb_examples)
    for i in range(nb_examples) :
        x = data[i, :]
        for j in range(dimension) :
            if int(np.floor(x[j] * grid_size)) % 2 != 0 :
                labels[i] = labels[i] * (-1)
        if np.random.rand() < noise :
            labels[i] = labels[i] * (-1)
    return data, labels


noises = [0, 0.1, 0.2]
score_max = 0
dima = []
nbcasea = []
ka = []
noisea = []
nbexa = []
scorea = []

for k in range(1, 6) :
    for dim in range(2, 11) :
        for nbcases in range(2, 9) :
            for noise in noises :
                for nbex in range(1000, 10000, 1000) :
                    X, y = damier(dim, nbcases, nbex, noise)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    neigh = KNeighborsClassifier(n_neighbors=k)
                    neigh.fit(X_train, y_train)
                    score = neigh.score(X_test, y_test)
                    if score_max < score:
                        score_max = score
                        print(k)
                        print(dim)
                        print(nbcases)
                        print(noise)
                        print(nbex)
                        print(score)
                        print('\n')

                    if score > 0.98:
                        ka.append(k)
                        dima.append(dim)
                        nbcasea.append(nbcases)
                        noisea.append(noise)
                        nbexa.append(nbex)
                        scorea.append(score)

plt.figure()
plt.title("k")
plt.scatter(ka, scorea)
plt.savefig('k.png')
plt.figure()
plt.title("dim")
plt.scatter(dima, scorea)
plt.savefig('dim.png')
plt.figure()
plt.title("nbcase")
plt.scatter(nbcasea, scorea)
plt.savefig('nbcase.png')
plt.figure()
plt.title("noise")
plt.scatter(noisea, scorea)
plt.savefig('noise.png')
plt.figure()
plt.title("nbex")
plt.scatter(nbexa, scorea)
plt.savefig('nb_ex.png')
plt.show()


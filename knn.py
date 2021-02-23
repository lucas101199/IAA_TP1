import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

"""
#Q1.1
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


# Q1.2
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


"""
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
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # separe les donnees en donnees train et test
                    neigh = KNeighborsClassifier(n_neighbors=k) cree un classifier pour les KNN
                    neigh.fit(X_train, y_train) # entraine le classifier avec les donnees d'entrainement
                    score = neigh.score(X_test, y_test) recupere le score du classifier sur les donnees test
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

best score k=4, dim=2, nbcase=2, noise=0, nbex=8000, score=0.99375


#Q1.2.2
X, y = damier(2, 2, 8000, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
score_max = 0
k_opt = 0
for k in range(1, 6):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=5) #cross validation sur le classifier
    if score_max < np.mean(scores):
        score_max = np.mean(scores)
        k_opt = k
        print(k_opt)
        print(score_max)

clf = KNeighborsClassifier(n_neighbors=k_opt)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


# Q1.2.3
for grid in range(2, 11):
    X, y = damier(2, grid, 1000, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    score_max = 0
    k_opt = 0
    for k in range(1, 6):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        if score_max < np.mean(scores):
            score_max = np.mean(scores)
            k_opt = k
    clf = KNeighborsClassifier(n_neighbors=k_opt)
    clf.fit(X_train, y_train)
    print("score = " + str(clf.score(X_test, y_test)))
    print("grid size = " + str(grid))
    print("k = " + str(k_opt))
"""

# Q1.2.4

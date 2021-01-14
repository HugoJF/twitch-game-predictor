import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing as preprocessing
import multiprocessing
from time import time
from joblib import Parallel, delayed
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import kmeans_plusplus
from matplotlib import pylab
from scipy import spatial
from multiprocessing import Pool, freeze_support, cpu_count

num_cores = multiprocessing.cpu_count()
# results = Parallel(n_jobs=num_cores)(delayed(extract_features)(i) for i in image_paths())

pylab.rcParams['figure.figsize'] = (16, 10)

games = ['apex', 'bf3', 'cod', 'eft', 'fortnite', 'gtav', 'lol', 'minecraft', 'overwatch', 'phasmo', 'raft',
         'rocketleague', 'rust', 'science', 'tab', 'thesims4']
images = 30
k = 1024*4  # kmeans++ cluster count

# feats = cv2.xfeatures2d.SIFT_create(2500)
feats = cv2.xfeatures2d.SURF_create(500, 6)
# feats = cv2.ORB_create(nfeatures=1500)

pool = list()


def image_path(game, i):
    return 'images/%s/%s.png' % (game, i)


def image_paths():
    imgs = list()

    for game in games:
        for i in range(0, images):
            imgs.append(image_path(game, i))

    return imgs


feature_cache = {}


def extract_features(path):
    global feature_cache

    if path not in feature_cache:
        start = time()
        img = cv2.imread(path)
        feature_cache[path] = feats.detectAndCompute(img, None)
        end = time()
        print('Extracting features for %s in %dms' % (path, (end - start) * 1000))
    else:
        print('Features cached for %s' % path)

    return feature_cache[path]


# Quickly check if every image exist before start extracting features
for path in image_paths():
    f = open(path)
    f.close()

feature_start = time()
for path in image_paths():
    keypoints, descriptors = extract_features(path)
    pool.extend(descriptors)
feature_end = time()
print('Feature extraction took %d secs' % (feature_end - feature_start))
print('Feature pool size: ', len(pool))

print('Clustering data...')
clustering_start = time()
centers_init, indices = kmeans_plusplus(np.array(pool), n_clusters=k, random_state=0)
clustering_end = time()
print('Clustering (k=%d) took %d secs' % (k, clustering_end - clustering_start))


def indices_to_centers(i):
    return pool[i]


centers = list(map(indices_to_centers, indices))

print('done')

tree = spatial.KDTree(centers)

# Carrega dataset
path = 'images/pred.png'
print(path)

x = list()
y = list()


def build_histogram(path, centers):
    keypoints, descriptors = extract_features(path)

    histo = [0] * len(centers)

    for desc in descriptors:
        coord, i = tree.query(desc)
        histo[i] = histo[i] + 1

    return histo


print('Building histograms...')
histo_start = time()
for game in games:
    for i in range(0, images):
        path = image_path(game, i)
        start = time()
        histo = build_histogram(path, centers)
        end = time()
        print('Histogram for image %d of %s took %dms' % (i, game, (end - start) * 1000))

        x.append(histo)
        y.append(game)
histo_end = time()
print('Histogram building took %d secs' % (histo_end - histo_start))

# xn = x
# x = preprocessing.normalize(x, norm='l1')


# Inicializa classificadores a serem utilizados
classifiers = {
    'knn':        KNeighborsClassifier(n_neighbors=4),
    'svm':        SVC(C=3),
    'nb':         GaussianNB(),
    'qda':        QuadraticDiscriminantAnalysis(),
    'tree':       DecisionTreeClassifier(max_depth=4),
    'regression': LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial', max_iter=500),
    'neural':     MLPClassifier(alpha=1, max_iter=1000),
    'gaussian':   GaussianProcessClassifier(1.0 * RBF(1.0)),
}

# Aumenta o tamanho das imagens (matrizes de confusão)
pylab.rcParams['figure.figsize'] = (10, 6)

npx = np.array(x)
npy = np.array(y)

# Itera classificadores
for name, classifier in classifiers.items():
    # Inicializa vetores dos resultados dos testes
    # Cada teste do LeaveOneOut será armazenado para poder gerar métricas em seguida
    trues = []
    preds = []

    # Iniciaza o LeaveOneOut
    loo = LeaveOneOut()

    # Realiza a divisão dos dados
    for train, test in loo.split(npx):
        # Separa parte de treino e parte de teste
        x_train, y_train = npx[train], npy[train]
        x_test, y_test = npx[test], npy[test]

        # Realiza o treino do classificador
        classifier.fit(x_train, y_train)

        # Prediz o teste com o classificador
        pred = classifier.predict(x_test)

        # Separa o valor real
        true = y_test

        # Adiciona na lista de predições
        # TODO: mudar para tuplas
        trues.append(true[0])
        preds.append(pred[0])

    print()
    print()
    print('#################################################')
    print('Estatísticas para %s' % name)
    print('Recall=', recall_score(trues, preds, average='weighted'))
    print('Precision=', precision_score(trues, preds, average='weighted'))
    print('Accuracy=', accuracy_score(trues, preds))

    # Lista de todos as classes sem repetição
    # TODO: isso faz sentido? nao perde a ordem?
    labels = list(set(trues))

    # Gera matriz de confusão
    matrix = confusion_matrix(trues, preds, labels=labels)

    # Inicia plot para matriz de confusão
    # ax = plt.subplot()
    # sns.heatmap(matrix, annot=True, ax=ax)  # annot=True to annotate cells

    # labels, title and ticks
    # ax.set_title('Confusion Matrix')

    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')

    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)

    # Printa matriz em texto
    print(matrix)
    print('#################################################')
    print()
    print()

    # Mostra matriz imagem
    # plt.show()

# Texto para ser testado
url = 'https://i.imgur.com/DT0zhfV.png'  # @param {type:"string"}
path = 'images/pred.png'

# !wget $url - O / content / drive / My\ Drive / Colab\ Notebooks / twitch - game - predictor / pred.png

# Itera classificadores
for name, model in classifiers.items():
    # Separa texto e intenção do dataset
    x = npx
    y = npy

    # Treina modelo
    model.fit(x, y)

    # Vetoriza teste
    test = build_histogram(path, centers)

    # Prediz
    pred = model.predict([test])

    # Printa predição
    print(name, '=', pred[0])

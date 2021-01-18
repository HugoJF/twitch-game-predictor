import cv2
import numpy as np
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib import pylab
from libKMCUDA import kmeans_cuda
from scipy import spatial
from time import time

# Increase size of figures
pylab.rcParams['figure.figsize'] = (16, 10)

# Classes
games = ['apex', 'bf3', 'cod', 'eft', 'fortnite', 'gtav', 'lol', 'minecraft', 'overwatch', 'phasmo', 'raft',
         'rocketleague', 'rust', 'science', 'tab', 'thesims4']

# Images to use in this
images = 30
k = 5 * 1024  # kmeans++ cluster count

# Feature descriptors
# feats = cv2.xfeatures2d.SIFT_create(2500)
feats = cv2.xfeatures2d.SURF_create(250, 7)
# feats = cv2.ORB_create(nfeatures=1500)

# Feature pool
pool = list()

# Memory feature cache [path]: [features]
feature_cache = {}


def image_path(game, i):
    return 'images/%s/%s.png' % (game, i)


def image_paths():
    imgs = list()

    for game in games:
        for i in range(0, images):
            imgs.append(image_path(game, i))

    return imgs


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


def build_histogram(path, centers):
    keypoints, descriptors = extract_features(path)

    histo = [0] * len(centers)

    for desc in descriptors:
        coord, i = tree.query(desc)
        histo[i] = histo[i] + 1

    return histo


def indices_to_centers(i):
    return pool[i]


def is_not_nan(n):
    c = np.sum(n)
    return not np.isnan(c)


# Quickly check if every image exist before start extracting features
for path in image_paths():
    f = open(path)
    f.close()


############################
# Feature extraction phase #
############################


print('Starting feature extraction...')
feature_start = time()

for path in image_paths():
    keypoints, descriptors = extract_features(path)
    pool.extend(descriptors)

feature_end = time()
print('Feature extraction took %d secs' % (feature_end - feature_start))

print('Feature pool size: ', len(pool))

####################
# Clustering phase #
####################


print('Clustering data...')
clustering_start = time()

# centers_init, indices = kmeans_plusplus(np.array(pool), n_clusters=k, random_state=0)
centers, assignments = kmeans_cuda(np.array(pool), clusters=k, verbosity=0, seed=0)

clustering_end = time()
print('Clustering (k=%d) took %d secs' % (k, clustering_end - clustering_start))

print('Centers before filtering: %d' % len(centers))
centers = list(filter(is_not_nan, centers))
print('Centers after filtering: %d' % len(centers))

###################
# Histogram phase #
###################


print('Building histograms...')
histo_start = time()
tree = spatial.KDTree(centers)

x = list()
y = list()

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


########################
# Classification phase #
########################


# Inicializa classificadores a serem utilizados
classifiers = {
    'svm':        SVC(C=3),
    'regression': LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial', max_iter=500),
    'neural':     MLPClassifier(alpha=1, max_iter=1000),
    'knn':        KNeighborsClassifier(n_neighbors=4),
}

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
    labels = list(set(trues))

    # Gera matriz de confusão
    matrix = confusion_matrix(trues, preds, labels=labels)

    # Inicia plot para matriz de confusão
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, ax=ax)  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Printa matriz em texto
    print(matrix)
    print('#################################################')
    print()
    print()

    # Mostra matriz imagem
    plt.show()

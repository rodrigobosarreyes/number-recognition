# MNIST
from mnist import MNIST
# Visualización de los datos
from matplotlib import pyplot as plt
# Trabajo con arrays
import numpy as np
# sklearn
from sklearn.model_selection import train_test_split
# Importamos el algoritmo de Decision Tree
from sklearn.tree import DecisionTreeClassifier
# sklearn
from sklearn.cluster import KMeans
# Matriz de confusión
from sklearn.metrics import confusion_matrix
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Generador
from sklearn.datasets.samples_generator import make_blobs


# Importamos nuestro set de datos
dataset = MNIST('MNIST_DATA')

# features => datos
# labels   => etiquetas
features, labels = dataset.load_training()

# matplotlib escala de grises
plt.gray()

# Mostramos los 25 primeros números
for i in range(25):
    plt.subplot(5,5,i+1)

    d_image = features[i]
    d_image = np.array(d_image, dtype='float')
    
    pixels = d_image.reshape((28, 28))
    
    plt.imshow(pixels, cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.show()

# todos los datos de training... deben ser divididos
# Datos de Entrenamiento 70%  (features, labels)
# Datos de Testing  30%   (features y labels)

# sklearn.model_selection.train_test_split(*arrays, **options)

# train_data, test_data, train_labels, test_labels
# 70% del total de los datos serán para el training set
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

clf_dt = DecisionTreeClassifier()

# Entrenamos al modelo
clf_dt.fit(train_data, train_labels)

# definimos una función para evaluar el clasificador
def evaluar_clasificador(clf, test_data, test_labels):
    pred = clf.predict(test_data)
    # Matriz de confusión
    # Mandamos los valores "reales" con los que predice
    MC = confusion_matrix(test_labels, pred)
    
    return MC

MC = evaluar_clasificador(clf_dt, test_data, test_labels)
print(MC)

# calcular el score
score = MC.diagonal().sum()*100/MC.sum()

print(score)

# Ahora lo hacemos con un RandomForest
# n_estimators = cuantidad de árboles
clf_rd = RandomForestClassifier(n_estimators=150, min_samples_split=2)

#Lo entrenamos
clf_rd.fit(train_data, train_labels)

MC = evaluar_clasificador(clf_rd, test_data, test_labels)
print(MC)
score = MC.diagonal().sum() * 100 / MC.sum()
print(score) 

# PRUEBAS
# Datos evaluación MNIST
test_data, test_labels = dataset.load_testing()

# Aplicamos el clasificador a todo el dataset de Evaluación y obtenemos el Accuracy
predicted = clf_rd.predict(test_data)

# Evaluamos el algoritmo
MC = evaluar_clasificador(clf_rd, test_data, np.array(test_labels))
print(MC)

# score
score = MC.diagonal().sum()*100/MC.sum()
print(score)

# Para visualizar
digito_extra = test_data[21]
d = np.array(digito_extra, dtype='float')
pixels = d.reshape((28,28))
plt.imshow(pixels, cmap='gray')
plt.show()

# aVer
print(clf_rd.predict([test_data[21]]))



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo CSV
df = pd.read_csv("dataset_final.csv", sep='|')

# Eliminar las columnas no necesarias
df = df.drop(columns=['Name', 'md5'])

# Separar las características y la etiqueta
X = df.drop(columns=['legitimate'])
y = df['legitimate']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar los clasificadores
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=50, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000),
    "Naïve Bayes": GaussianNB()
}

# Entrenar los clasificadores y generar las matrices de confusión
fig, axes = plt.subplots(3, 2, figsize=(12, 18))  # 3 filas, 2 columnas
axes = axes.ravel()  # Aplanar la matriz de ejes para un fácil acceso

for idx, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Visualizar la matriz de confusión
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=clf.classes_)
    disp.plot(ax=axes[idx], values_format='d')
    axes[idx].set_title(name)

# Ajustar el diseño
plt.tight_layout()
plt.show()

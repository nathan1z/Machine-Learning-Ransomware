import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn import model_selection
from sklearn import svm, model_selection
from sklearn.metrics import f1_score

# Cargar el archivo CSV
df = pd.read_csv("dataset_final.csv", sep='|')
y = df['legitimate']

# Excluir las columnas 'Name', 'md5', y 'legitimate'
df_features = df.drop(columns=['Name', 'md5', 'legitimate'])

# Estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
print(X_scaled.shape)

# Aplicar PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)
print(pca.explained_variance_ratio_)

# Dividir datos en datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Imprima el número de muestras de entrenamiento y prueba
print("\n\t[*] Ejemplos de entrenamiento: ", len(X_train))
print("\t[*] Muestras de prueba: ", len(X_test))

# Entrene el algoritmo de SVM en el conjunto de datos de entrenamiento
clf = svm.SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Predecir las clases del conjunto de pruebas
y_pred = clf.predict(X_test)

# Imprime la precisión del modelo
print("\n\t[*] Exactitud:", clf.score(X_test, y_test))

# Realice una validación cruzada e imprima la precisión
score = model_selection.cross_val_score(clf, X_pca, y, cv=10)  # Usar X_pca y y completos
print("\n\t[*] Puntuación de validación cruzada: ", round(score.mean() * 100, 2), '%')

# Calcular la puntuación F1
f = f1_score(y_test, y_pred)  # Usar los datos de prueba para la puntuación F1
print("\t[*] Puntuación F1: ", round(f * 100, 2), '%')
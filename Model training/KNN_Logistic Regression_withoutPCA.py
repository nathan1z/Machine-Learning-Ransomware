import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Cargar el archivo CSV
df = pd.read_csv("dataset_final.csv", sep='|')
y = df['legitimate']

# Excluir las columnas 'Name', 'md5', y 'legitimate'
df_features = df.drop(columns=['Name', 'md5', 'legitimate'])

# Estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
print(X_scaled.shape)

# Dividir datos en datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Imprimir el número de muestras de entrenamiento y prueba
print("\n\t[*] Ejemplos de entrenamiento: ", len(X_train))
print("\t[*] Muestras de prueba: ", len(X_test))

# Entrenar el algoritmo de k-NN en el conjunto de datos de entrenamiento
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predecir las clases del conjunto de pruebas
y_pred_knn = knn.predict(X_test)

# Imprimir la precisión del modelo k-NN
print("\n\t[*] Exactitud (k-NN):", knn.score(X_test, y_test))

# Realizar una validación cruzada e imprimir la precisión para k-NN
score_knn = model_selection.cross_val_score(knn, X_scaled, y, cv=10)  # Usar X_scaled y y completos
print("\n\t[*] Puntuación de validación cruzada (k-NN): ", round(score_knn.mean() * 100, 6), '%')

# Calcular la puntuación F1 para k-NN
f_knn = f1_score(y_test, y_pred_knn)  # Usar los datos de prueba para la puntuación F1
print("\t[*] Puntuación F1 (k-NN): ", round(f_knn * 100, 6), '%')

# Entrenar el algoritmo de Regresión Logística en el conjunto de datos de entrenamiento
clf_lr = LogisticRegression(random_state=42, max_iter=1000)
clf_lr.fit(X_train, y_train)

# Predecir las clases del conjunto de pruebas
y_pred_lr = clf_lr.predict(X_test)

# Imprimir la precisión del modelo de Regresión Logística
print("\n\t[*] Exactitud (Logistic Regression):", clf_lr.score(X_test, y_test))

# Realizar una validación cruzada e imprimir la precisión para Regresión Logística
score_lr = model_selection.cross_val_score(clf_lr, X_scaled, y, cv=10)  # Usar X_scaled y y completos
print("\n\t[*] Puntuación de validación cruzada (Logistic Regression): ", round(score_lr.mean() * 100, 6), '%')

# Calcular la puntuación F1 para Regresión Logística
f_lr = f1_score(y_test, y_pred_lr)  # Usar los datos de prueba para la puntuación F1
print("\t[*] Puntuación F1 (Logistic Regression): ", round(f_lr * 100, 6), '%')

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Cargar los datos desde un archivo CSV
df = pd.read_csv("dataset_final.csv", sep='|')

# Eliminar las columnas no necesarias
df = df.drop(columns=['Name', 'md5'])

# Separar las características y la etiqueta
X = df.drop(columns=['legitimate'])
y = df['legitimate']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el clasificador SVM
svm_classifier = SVC(kernel='linear', random_state=42)  # Puedes probar otros kernels como 'rbf'

# Entrenar el clasificador
svm_classifier.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.6f}')
print(f'F1 Score: {f1:.6f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

# Validación cruzada
cv_scores = cross_val_score(svm_classifier, X, y, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Accuracy: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}')

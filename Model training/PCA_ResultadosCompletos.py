import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv("dataset_final.csv", sep='|')
y = df['legitimate']

# Excluir las columnas 'Name', 'md5', y 'legitimate'
df_features = df.drop(columns=['Name', 'md5', 'legitimate'])

# Estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Aplicar PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Inicializar los clasificadores
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=50, random_state=42)
}

# Evaluar los clasificadores
results = []

for name, clf in classifiers.items():
    # Validación cruzada
    cv_scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    
    # Entrenamiento y predicción
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Métricas
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Cross-Validation Accuracy": np.mean(cv_scores),
        "F1 Score": f1,
        "Accuracy": accuracy
    })

# Mostrar resultados
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_2/data_limpia.csv")

deps = df['dependencia'].value_counts().index[:3]

# Usar solo 3 dependencias para clasificación
df_knn = df[df['dependencia'].isin(deps)]

X = df_knn[['mes', 'anio']]
y = df_knn['dependencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Precisión del modelo KNN:", knn.score(X_test, y_test))
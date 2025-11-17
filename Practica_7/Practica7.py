import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Cargar datos
df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_7/data_limpia.csv")

#Usar dos columnas numéricas para agrupar
X = df[["Sueldo Neto", "mes"]]

#K-Means con 3 grupos
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

#Agregar etiqueta de cluster y “probarlo” viendo métricas simples
df["cluster"] = kmeans.labels_
print("Elementos por cluster:")
print(df["cluster"].value_counts().sort_index())
print("Inercia (qué tan compactos quedan los grupos):", kmeans.inertia_)

#Gráfica simple (muestra 1000 puntos para que no pese)
muestra = df.sample(min(1000, len(df)), random_state=0)
plt.scatter(muestra["Sueldo Neto"], muestra["mes"], c=muestra["cluster"], alpha=0.5)
plt.xlabel("Sueldo Neto")
plt.ylabel("Mes")
plt.title("K-Means (k=3)")
plt.show()

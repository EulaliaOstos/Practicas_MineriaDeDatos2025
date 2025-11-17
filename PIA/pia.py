import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, kruskal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud            
from sklearn.cluster import KMeans  

#Practica 1 (Data Cleaning)
df = pd.read_csv("uanl_02_2024.csv")

df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
df["Sueldo Neto"] = pd.to_numeric(df["Sueldo Neto"], errors="coerce")

df["fecha"] = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1), errors="coerce")

df["dependencia"] = df["dependencia"].astype(str).str.strip().str.upper()
df["Nombre"] = df["Nombre"].astype(str).str.strip()

df = df[df["mes"].between(1, 12)]
df = df[df["Sueldo Neto"].ge(0) | df["Sueldo Neto"].isna()]

df = df.dropna(subset=["anio", "mes", "fecha"])
df = df.drop_duplicates()

df.to_csv("data_limpia.csv", index=False)

print("✅ Práctica 1 terminada, archivo data_limpia.csv creado")

#Practica 2 (Descriptive Statistics)
print("\nEstadísticas descriptivas:")
print(df.describe())
print("\nPromedio de Sueldo Neto por año:")
print(df.groupby("anio")["Sueldo Neto"].mean())
print("\nPromedio de Sueldo Neto por mes:")
print(df.groupby("mes")["Sueldo Neto"].mean())

df["Sueldo Neto"].hist()
plt.show()
df.groupby("anio")["Sueldo Neto"].mean().plot(kind="bar")
plt.show()
df.groupby("mes")["Sueldo Neto"].mean().plot(kind="bar")
plt.show()

#Practica 3 (Data Visualization)
df["Sueldo Neto"].hist(bins=30)
plt.title("Histograma de Sueldo Neto")
plt.xlabel("Sueldo Neto")
plt.ylabel("Frecuencia")
plt.show()

df.boxplot(column="Sueldo Neto", by="anio")
plt.title("Boxplot de Sueldo Neto por Año")
plt.suptitle("")
plt.show()

df.groupby("mes")["Sueldo Neto"].mean().plot(kind="bar")
plt.title("Promedio de Sueldo Neto por Mes")
plt.xlabel("Mes")
plt.ylabel("Promedio")
plt.show()

df["anio"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Distribución de registros por Año")
plt.ylabel("")
plt.show()

df.plot.scatter(x="mes", y="Sueldo Neto")
plt.title("Sueldo Neto vs Mes")
plt.show()

#Practica 4 (Static Test)
deps = df['dependencia'].value_counts().index[:3]

samples = [
    df[df['dependencia'] == dep]['Sueldo Neto'].sample(100, random_state=1)
    for dep in deps
]

print("\nANOVA (compara los 3 grupos)")
print(f_oneway(*samples))

print("\nPrueba T (solo entre los dos primeros grupos)")
print(ttest_ind(samples[0], samples[1]))

print("\nKruskal–Wallis (no paramétrica, 3 grupos)")
print(kruskal(*samples))

#Practica 5 (Linear Models + correlation)
df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})

df['tiempo'] = df['anio'] * 12 + df['mes']
X = df[['tiempo']]
y = df['Sueldo Neto']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("R2 Score:", r2_score(y, y_pred))

plt.figure()
plt.scatter(X, y, alpha=0.1, label='Datos')
plt.plot(X, y_pred, color='red', label='Modelo lineal')
plt.title("Regresión lineal: Sueldo vs Tiempo")
plt.legend()
plt.show()

#Practica 6 (Data Classification)
deps = df['dependencia'].value_counts().index[:3]

df_knn = df[df['dependencia'].isin(deps)]

X = df_knn[['mes', 'anio']]
y = df_knn['dependencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Precisión del modelo KNN:", knn.score(X_test, y_test))

#Practica 7 (Data Clustering)
X = df[["Sueldo Neto", "mes"]]

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

df["cluster"] = kmeans.labels_
print("Elementos por cluster:")
print(df["cluster"].value_counts().sort_index())
print("Inercia (qué tan compactos quedan los grupos):", kmeans.inertia_)

muestra = df.sample(min(1000, len(df)), random_state=0)
plt.scatter(muestra["Sueldo Neto"], muestra["mes"], c=muestra["cluster"], alpha=0.5)
plt.xlabel("Sueldo Neto")
plt.ylabel("Mes")
plt.title("K-Means (k=3)")
plt.show()

#Practica 8 (Forecasting)

df["fecha"] = pd.to_datetime({"year": df["anio"], "month": df["mes"], "day": 1})
df = df.sort_values("fecha")

promedio_mensual = df.groupby("fecha")["Sueldo Neto"].mean().reset_index()

X = np.arange(len(promedio_mensual)).reshape(-1, 1)
y = promedio_mensual["Sueldo Neto"].values

modelo = LinearRegression()
modelo.fit(X, y)

futuro_X = np.arange(len(promedio_mensual), len(promedio_mensual) + 12).reshape(-1, 1)
futuro_y = modelo.predict(futuro_X)
futuras_fechas = pd.date_range(start=promedio_mensual["fecha"].iloc[-1] + pd.offsets.MonthBegin(1),
                               periods=12, freq="MS")

print("Pronóstico de sueldo promedio para los próximos 12 meses:")
for i, valor in enumerate(futuro_y, 1):
    print(f"Mes {i}: {valor:.2f}")

plt.figure()
plt.plot(promedio_mensual["fecha"], y, label="Histórico")
plt.plot(futuras_fechas, futuro_y, "--", label="Pronóstico", color="red")
plt.title("Pronóstico del Sueldo Promedio con Regresión Lineal")
plt.xlabel("Fecha")
plt.ylabel("Sueldo Promedio")
plt.legend()
plt.show()

#Practica 9 (Text analysis)

text = " ".join(df['Nombre'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de palabras de nombres")
plt.show()
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway, ttest_ind, kruskal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud            
from sklearn.cluster import KMeans  

# Practica 1 (Data Cleaning)
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

# Practica 2 (Descriptive Statistics)
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

# Practica 3 (Data Visualization)
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

# Practica 4 (Static Test)
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

# Practica 5 (Linear Models + correlation)
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

# Practica 6 (Data Classification)
deps = df['dependencia'].value_counts().index[:3]

df_knn = df[df['dependencia'].isin(deps)]

X = df_knn[['mes', 'anio']]
y = df_knn['dependencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Precisión del modelo KNN:", knn.score(X_test, y_test))

# Practica 7 (Data Clustering)
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

# Practica 8 (Forecasting)
df["fecha"] = pd.to_datetime({"year": df["anio"], "month": df["mes"], "day": 1})
df = df.sort_values("fecha")

promedio_mensual = df.groupby("fecha")["Sueldo Neto"].mean().reset_index()

X = np.arange(len(promedio_mensual)).reshape(-1, 1)
y = promedio_mensual["Sueldo Neto"].values

modelo = LinearRegression()
modelo.fit(X, y)

futuro_X = np.arange(len(promedio_mensual), len(promedio_mensual) + 12).reshape(-1, 1)
futuro_y = modelo.predict(futuro_X)
futuras_fechas = pd.date_range(
    start=promedio_mensual["fecha"].iloc[-1] + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

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

# Practica 9 (Text analysis)
text = " ".join(df['Nombre'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de palabras de nombres")
plt.show()

# SEGUNDAS 
print("\n" + "="*60)
print("ANÁLISIS ADICIONAL DEL PIA (SEGUNDAS)")
print("="*60)

# ----- HIPÓTESIS 1: ¿Pagan igual PREPARATORIAS y FACULTADES? -----
prepas = df[df["dependencia"].str.contains("PREPARATORIA", case=False, na=False)]
facs   = df[df["dependencia"].str.contains("FACULTAD", case=False, na=False)]

# Nos quedamos solo con sueldos válidos
sueldos_prepas = prepas["Sueldo Neto"].dropna()
sueldos_facs   = facs["Sueldo Neto"].dropna()

t_stat, p_val = ttest_ind(
    sueldos_prepas,
    sueldos_facs,
    equal_var=False
)

print("\nHIPÓTESIS 1: Sueldo en PREPARATORIAS vs FACULTADES")
print("Tamaño muestra PREPAS:     ", len(sueldos_prepas))
print("Tamaño muestra FACULTADES:", len(sueldos_facs))
print("Promedio PREPAS:          ", sueldos_prepas.mean())
print("Promedio FACULTADES:      ", sueldos_facs.mean())
print("p-value prueba t:", p_val)

if p_val < 0.05:
    print("Conclusión: Sí hay diferencia significativa entre sueldos de prepas y facultades.")
else:
    print("Conclusión: No se encontró diferencia significativa entre sueldos de prepas y facultades.")

# --- GRÁFICA: Comparación de sueldos promedio PREPAS vs FACULTADES ---
plt.figure(figsize=(6,4))
plt.bar(["Prepas", "Facultades"], 
        [sueldos_prepas.mean(), sueldos_facs.mean()],
        color=["#4C72B0", "#55A868"])
plt.title("Comparación de sueldo promedio\nPreparatorias vs Facultades")
plt.ylabel("Sueldo promedio")
plt.show()

# ----- HIPÓTESIS 2: ¿Todas las PREPAS pagan igual? (ANOVA) -----
prepas_todas = df[df["dependencia"].str.contains("ESCUELA PREPARATORIA", case=False, na=False)]

grupos = []
nombres = []

for nombre, grupo in prepas_todas.groupby("dependencia"):
    # Para que no meta grupos con muy poquitos registros
    if len(grupo) > 30:
        grupos.append(grupo["Sueldo Neto"].dropna().values)
        nombres.append(nombre)

if len(grupos) >= 2:
    f_stat, p_val_anova = f_oneway(*grupos)
    print("\nHIPÓTESIS 2: Comparación de sueldos entre ESCUELAS PREPARATORIAS")
    print("Preparatorias incluidas:", nombres)
    print("p-value ANOVA:", p_val_anova)
    print("\nPromedio de sueldo por preparatoria:")
    print(prepas_todas.groupby("dependencia")["Sueldo Neto"].mean().sort_values())

    if p_val_anova < 0.05:
        print("Conclusión: No todas las prepas pagan igual (hay diferencias significativas).")
    else:
        print("Conclusión: No se encontraron diferencias significativas entre las prepas.")

    # --- PRUEBA POST-HOC: TUKEY PARA SABER QUÉ PREPAS SON DIFERENTES ---
    print("\nAnálisis post-hoc (Tukey HSD) entre preparatorias:")

    tukey = pairwise_tukeyhsd(
        endog=prepas_todas["Sueldo Neto"].dropna(),
        groups=prepas_todas["dependencia"].dropna(),
        alpha=0.05
    )

    print(tukey.summary())
else:
    print("\nNo hay suficientes prepas con datos para hacer ANOVA.")

# --- GRÁFICA: Sueldo promedio por Preparatoria ---
prom_prepas = prepas_todas.groupby("dependencia")["Sueldo Neto"].mean().sort_values()

plt.figure(figsize=(12,6))
prom_prepas.plot(kind="bar", color="#4C72B0")
plt.title("Sueldo promedio por Preparatoria")
plt.ylabel("Sueldo promedio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ----- HIPÓTESIS 3: ¿Todas las FACULTADES pagan igual? (ANOVA) -----
facs_todas = df[df["dependencia"].str.contains("FAC.", case=False, na=False)]

grupos_f = []
nombres_f = []

for nombre, grupo in facs_todas.groupby("dependencia"):
    if len(grupo) > 30:
        grupos_f.append(grupo["Sueldo Neto"].dropna().values)
        nombres_f.append(nombre)

if len(grupos_f) >= 2:
    f_stat_f, p_val_anova_f = f_oneway(*grupos_f)
    print("\nHIPÓTESIS 3: Comparación de sueldos entre FACULTADES")
    print("Facultades incluidas:", nombres_f)
    print("p-value ANOVA:", p_val_anova_f)
    print("\nPromedio de sueldo por facultad:")
    print(facs_todas.groupby("dependencia")["Sueldo Neto"].mean().sort_values())

    if p_val_anova_f < 0.05:
        print("Conclusión: No todas las facultades pagan igual (hay diferencias significativas).")
    else:
        print("Conclusión: No se encontraron diferencias significativas entre las facultades.")
else:
    print("\nNo hay suficientes facultades con datos para hacer ANOVA.")

# --- GRÁFICA: Sueldo promedio por Facultad ---
prom_facs = facs_todas.groupby("dependencia")["Sueldo Neto"].mean().sort_values()

plt.figure(figsize=(12,6))
prom_facs.plot(kind="bar", color="#55A868")
plt.title("Sueldo promedio por Facultad")
plt.ylabel("Sueldo promedio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ----- CLUSTERING: Dependencias según su sueldo promedio -----
prom_dep = df.groupby("dependencia", as_index=False)["Sueldo Neto"].mean()

kmeans_dep = KMeans(n_clusters=3, random_state=0)
prom_dep["cluster"] = kmeans_dep.fit_predict(prom_dep[["Sueldo Neto"]])

print("\nCLUSTERING de dependencias por sueldo promedio")
for c in sorted(prom_dep["cluster"].unique()):
    print(f"\nCluster {c} (dependencias con sueldo promedio similar):")
    tmp = prom_dep[prom_dep["cluster"] == c].sort_values("Sueldo Neto")
    print(tmp.head(10))  
# --- GRÁFICA: Clustering de dependencias por sueldo promedio ---
plt.figure(figsize=(7,5))
plt.scatter(prom_dep["Sueldo Neto"], prom_dep["cluster"], c=prom_dep["cluster"])
plt.title("Clustering de dependencias según sueldo promedio")
plt.xlabel("Sueldo promedio")
plt.ylabel("Cluster")
plt.show()


# ----- CLUSTERING AVANZADO: promedio, desviación estándar y cantidad de empleados -----
stats_dep = df.groupby("dependencia")["Sueldo Neto"].agg(["mean", "std", "count"]).dropna()

kmeans_adv = KMeans(n_clusters=3, random_state=0)
stats_dep["cluster"] = kmeans_adv.fit_predict(stats_dep[["mean", "std", "count"]])

print("\nCLUSTERING AVANZADO de dependencias (mean, std, count)")
for c in sorted(stats_dep["cluster"].unique()):
    print(f"\nCluster {c}:")
    print(stats_dep[stats_dep["cluster"] == c].sort_values("mean").head(10))
# --- GRÁFICA: Clustering avanzado en 2D (mean vs std) ---
plt.figure(figsize=(7,5))
plt.scatter(stats_dep["mean"], stats_dep["std"], c=stats_dep["cluster"])
plt.title("Clustering avanzado: mean vs std de sueldo por dependencia")
plt.xlabel("Sueldo promedio")
plt.ylabel("Desviación estándar")
plt.show()


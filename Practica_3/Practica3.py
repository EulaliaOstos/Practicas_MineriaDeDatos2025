import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_3/data_limpia.csv")

# Histograma de sueldos
df["Sueldo Neto"].hist(bins=30)
plt.title("Histograma de Sueldo Neto")
plt.xlabel("Sueldo Neto")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot de sueldos por año
df.boxplot(column="Sueldo Neto", by="anio")
plt.title("Boxplot de Sueldo Neto por Año")
plt.suptitle("")
plt.show()

# Gráfico de barras promedio por mes
df.groupby("mes")["Sueldo Neto"].mean().plot(kind="bar")
plt.title("Promedio de Sueldo Neto por Mes")
plt.xlabel("Mes")
plt.ylabel("Promedio")
plt.show()

# Pie chart de cantidad de registros por año
df["anio"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Distribución de registros por Año")
plt.ylabel("")
plt.show()

# Scatter de Sueldo Neto vs Mes
df.plot.scatter(x="mes", y="Sueldo Neto")
plt.title("Sueldo Neto vs Mes")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_2/data_limpia.csv")


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

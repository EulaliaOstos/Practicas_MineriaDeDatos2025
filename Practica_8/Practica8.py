import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar los datos
df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_8/data_limpia.csv")

#  Crear la columna de fecha si no está
df["fecha"] = pd.to_datetime({"year": df["anio"], "month": df["mes"], "day": 1})
df = df.sort_values("fecha")

# Agrupar por fecha y calcular el sueldo promedio
promedio_mensual = df.groupby("fecha")["Sueldo Neto"].mean().reset_index()

# Preparar datos para el modelo
X = np.arange(len(promedio_mensual)).reshape(-1, 1)
y = promedio_mensual["Sueldo Neto"].values

# Entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predecir 12 meses futuros
futuro_X = np.arange(len(promedio_mensual), len(promedio_mensual) + 12).reshape(-1, 1)
futuro_y = modelo.predict(futuro_X)
futuras_fechas = pd.date_range(start=promedio_mensual["fecha"].iloc[-1] + pd.offsets.MonthBegin(1),
                               periods=12, freq="MS")

# Mostrar resultados
print("Pronóstico de sueldo promedio para los próximos 12 meses:")
for i, valor in enumerate(futuro_y, 1):
    print(f"Mes {i}: {valor:.2f}")

#  Graficar los resultados
plt.figure()
plt.plot(promedio_mensual["fecha"], y, label="Histórico")
plt.plot(futuras_fechas, futuro_y, "--", label="Pronóstico", color="red")
plt.title("Pronóstico del Sueldo Promedio con Regresión Lineal")
plt.xlabel("Fecha")
plt.ylabel("Sueldo Promedio")
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_2/data_limpia.csv")

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

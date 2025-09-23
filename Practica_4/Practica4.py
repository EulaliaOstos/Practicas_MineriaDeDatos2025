import pandas as pd
from scipy.stats import f_oneway, ttest_ind, kruskal


df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_2/data_limpia.csv")

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

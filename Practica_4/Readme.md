# Práctica 4 – Pruebas estadísticas

Con el archivo `data_limpia.csv` quise ver si había diferencia en los sueldos entre las dependencias con más registros.  
Primero saqué tres grupos de dependencias y tomé 100 sueldos de cada uno para comparar.

Hice tres pruebas:
- **ANOVA** para revisar si al menos un grupo tiene un promedio de sueldo distinto.
- **Prueba T** para comparar solo los dos primeros grupos.
- **Kruskal-Wallis** que sirve aunque los datos no sean normales.

En las tres el p-valor me salió muy chiquito (mucho menor a 0.05), así que sí hay diferencia en los sueldos de esos grupos.
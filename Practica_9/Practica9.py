import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("/Users/gloria/Desktop/TAREAS FACU/7mo Semestre/Mineria de datos /Practicas/Practica_8/data_limpia.csv")

# Crear WordCloud de los nombres
text = " ".join(df['Nombre'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de palabras de nombres")
plt.show()
import pandas as pd


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


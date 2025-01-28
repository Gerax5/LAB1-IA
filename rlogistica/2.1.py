#esto es lo mismo que el jupyter de rl2, pero en python, porque jupyter habia dado varios errores, pero luego se soluciono al reiniarlo 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from RL import LogisticRegression
import matplotlib.pyplot as plt
# from sklearn.cluster import Kmeans

# Cargar los datos
df = pd.read_csv("dataset_phishing.csv")

# Mostrar las columnas
print("mostrar las columnas")
print(list(df.columns))

# Para saber si el dataset está balanceado
print("- - - - - - - BALANCEO - - - - - - -")
print(df["status"].value_counts())

# Obtener tipos de datos
tipos_de_datos = df.dtypes.value_counts()
print("- - - - - - - TIPOS DE DATOS - - - - - - -")
print(tipos_de_datos)

#verificar los valores nulos
print("verificar los valores nulos")
print(df.isnull().sum())

#informacion general del dataframe
print("informacion general del dataframe")
print(df.info())

#--eliminar los valores nulos o llenarlos con la media--
#eliminar filas con valores nulos
df = df.dropna()

#calcular la media solo de las columnas numericas
mean_values = df.select_dtypes(include=[np.number]).mean()

#llenar los valores nulos en las columnas numéricas
df.fillna(mean_values, inplace=True)

#eliminar duplicados
df = df.drop_duplicates()



#--convertir los tipos de datos----
# Convirtiendo Columna status a numerica
df["status"] = df["status"].map({"phishing": 1, "legitimate": 0})

# Columna innecesaria porque ya se tiene su longitud
dfProcessed = df.drop('url', axis=1)
print(list(dfProcessed.columns))
nulos_por_columna = df.isnull().sum()
print(list(nulos_por_columna))



#--normalizacion de datos 
#nombre de las columnas numericas a normalizar
numeric_columns = dfProcessed.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
dfProcessed[numeric_columns] = scaler.fit_transform(dfProcessed[numeric_columns])

print(dfProcessed.head())


#---Crear conjuntos de entrenamiento y prueba
X = dfProcessed.drop('status', axis=1)
y = dfProcessed['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predecir sobre los datos de prueba
predicciones = model.predict(X_test)
print('predicciones:')
print(predicciones)


# Asegurarse de que y_test sea un vector 1D y tenga valores 0 y 1
y_test = y_test.to_numpy().flatten()
y_test = np.where(y_test == -1, 0, y_test)  # Convertir -1 a 0 


#metricas de desempeño
# Calcular la precisión del modelo
print("metricas de desempeño:")

precision = precision_score(y_test, predicciones)
print("Precision:", precision)

accuracy = accuracy_score(y_test, predicciones)
print('accuracy (Exactitud):: ', accuracy)
recall= recall_score(y_test, predicciones)
print('recall (Sensibilidad): ', recall)

f1= f1_score(y_test, predicciones)
print('f1 score: ', f1)
print('el valor de f1 al ser cercano a 1 indica un buen equilibrio entre la presicion y el recall, entonces el modelo tiene un buen rendimiento en estas metricas')

matriz_de_confusion= confusion_matrix(y_test, predicciones)
print(f"Confusion Matrix:\n {matriz_de_confusion}")



#---graficar los grupos encontrados 

test_y = np.array(y_test)
test_x = X_test

y_pred= np.array(predicciones)

plt.figure(figsize=(10, 6))

# Graficar puntos correctamente clasificados
for label, marker, color in zip([0, 1], ['o', 'x'], ['blue', 'red']):
    plt.scatter(
        test_x.loc[(test_y == label) & (y_pred == label), 'length_url'],
        test_x.loc[(test_y == label) & (y_pred == label), 'length_hostname'],
        label=f'Correcto {label}',
        marker=marker,
        color=color,
        s=100,
        alpha=0.7
    )

# Graficar puntos mal clasificados
plt.scatter(
    test_x.loc[test_y != y_pred, 'length_url'],
    test_x.loc[test_y != y_pred, 'length_hostname'],
    label='Mal clasificados',
    marker='x',
    color='orange',
    s=100,
    alpha=0.7
)


plt.title("Predicción de regresion logistica: Longitud del URL vs Longitud del Hostname", fontsize=14)
plt.xlabel("Longitud del URL", fontsize=12)
plt.ylabel("Longitud del Hostname", fontsize=12)
plt.legend(title="Clases", fontsize=10)
plt.grid(True)
plt.show()




#prediccion Longitud del URL vs Longitud del Hostname
plt.figure(figsize=(10, 6))

for label, marker, color in zip([0, 1], ['o', 'x'], ['blue', 'red']):
    plt.scatter(
        test_x.loc[y_test == label, 'length_url'],
        test_x.loc[y_test == label, 'length_hostname'],
        label=f'Real {label}',
        marker=marker,
        color=color,
        s=100,
        alpha=0.7
    )

plt.title("Predicción de regresion logistica: Longitud del URL vs Longitud del Hostname", fontsize=14)
plt.xlabel("Longitud del URL", fontsize=12)
plt.ylabel("Longitud del Hostname", fontsize=12)
plt.legend(title="Status", fontsize=10)
plt.grid(True)
plt.show()
# Task 1 - Regresión Lineal

Considera un modelo de regresión lineal con dos características, X₁ y X₂, y sus pesos correspondientes w₁ y 
w₂. Si el modelo predice una salida y mediante la ecuación y = 2w₁X₁ + 3w₂X₂ + 1, ¿cuál es la interpretación 
del coeficiente 3w₂ en el contexto del modelo? 

    -----------------




-----------
Explica el concepto de multicolinealidad en el contexto de la regresión lineal. ¿Cómo afecta la 
multicolinealidad a la interpretación de los coeficientes de regresión individuales?

    La multicolinealidad se crea cuando dos o más varibles independientes estan altamente correlacionadas entre sí. Entonces, una variable puede ser predicha a una gran medida a partir de las otras. Sin embargo, esto puede causar problemas en la estimación de los coeficientes de regresión, siendo más dificil de interpretar y más inestables los coeficientes. 

    La manera que afecta la multicolinealidad a la interpretación de los coeficientes individuales: 
    - Inestabilidad de los coeficientes, esto puede significar que pequeños cambios en los datos pueden resultar variaciones grandes en los coeficientes estimados.

    - Significancia estadística, la multicolinealidad puede hacer que los coeficientes de las variables se vean que no son significativos, teniendo valores p altos, a pesar que si estas variables tengan una relación real con la variable dependiente. Esto se da porque la varianza de los coeficientes aumenta, y reduce la precisión de los resultados.  

    - Errores estándar elevados, esto se debe a que pueden ser muy grandes por la multicolianealidad, llevando a los intervalos de confianza amplios y esto dificulta la determinación precisa del efecto de cada variable dependiente. 
    
    - Dificultad de interpretación, por sus efectos mezclados por la alta correlación entre ellas es muy difícil interpretar el efecto individual de cada variable independiente. Entonces, es complicado determinar la variación en la variable dependiente se debe a una variable independiente especifica. 


    Para detectar la multicolinealildad existen diferentes técnicas, como la del Factor de Inflación de la varianza y el análisis de correlación entre variables independientes. 

    Mientras que para manejar la multicolienalidad lo recomendado es: eliminar variables que esten altamente correlacionadas, regresión Ridge y el análisis de componentes principales. 
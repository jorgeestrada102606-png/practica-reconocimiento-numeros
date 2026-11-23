# 📝 Práctica: Reconocimiento de Números Escritos a Mano
**Materia:** Fundamentos de Inteligencia Artificial  
**Modalidad:** Individual, en Google Colab / Jupyter  
**Datos:** 1,797 imágenes de números del 0 al 9 (8×8 píxeles = 64 características) de `scikit-learn`  
**Modelos:** KNN, SVM y Red Neuronal (MLP)  

---

## 👤 Datos del Alumno
- **Nombre:** ______________________________
- **Grupo:** ________
- **Fecha:** 23 de noviembre de 2026

---

## 📊 Tabla Resumen de Resultados Experimentales

| Modelo | Exactitud Entrenamiento | Exactitud Prueba | Errores (de 450) | Comportamiento |
| :--- | :---: | :---: | :---: | :--- |
| **Línea Base (Dummy)** | - | 0.102 (10.2%) | 404 | Adivina la clase más frecuente sin ver la imagen |
| **KNN (k=3)** | 0.996 (99.6%) | 0.984 (98.4%) | 7 | Vota con las 3 imágenes más cercanas |
| **SVM (SVC)** | **0.999 (99.9%)** | **0.991 (99.1%)** | **4** | **Mejor modelo general, margen máximo** |
| **Red Neuronal (MLP)** | **1.000 (100.0%)** | 0.962 (96.2%) | 17 | **Presenta sobreajuste (overfitting)** |

---

## ✍️ Respuestas a las Preguntas de la Práctica

### **Pregunta 1.** ¿Cuántas imágenes hay de cada número? ¿Están equilibradas? ¿Qué números crees que serán más difíciles de distinguir entre sí?

#### **1. Conteo de imágenes por dígito:**
- Dígito **0**: 178 imágenes
- Dígito **1**: 182 imágenes
- Dígito **2**: 177 imágenes
- Dígito **3**: 183 imágenes
- Dígito **4**: 181 imágenes
- Dígito **5**: 182 imágenes
- Dígito **6**: 181 imágenes
- Dígito **7**: 179 imágenes
- Dígito **8**: 174 imágenes
- Dígito **9**: 180 imágenes
- **Total:** 1,797 imágenes.

#### **2. Análisis de equilibrio:**
**Sí, el conjunto está sumamente equilibrado.** Cada dígito cuenta con aproximadamente 180 imágenes, lo que representa alrededor del 10% del total para cada clase. La variación entre la clase con menos muestras (el 8 con 174) y la de más muestras (el 3 con 183) es de apenas 9 imágenes (menos del 5% de diferencia relativa). No existe sesgo hacia ningún número en particular.

#### **3. Números más difíciles de distinguir:**
- **1 y 7:** Ambos dígitos tienen como eje principal una línea vertical descendente en el centro. Al estar en una resolución de solo 8×8 píxeles, si la barra superior horizontal del 7 es corta o tenue, la matriz se reduce casi a la misma columna de píxeles oscuros que un 1.
- **3 y 8:** Ambos comparten dos lóbulos redondeados hacia el costado derecho. Si una persona escribe un 8 sin cerrar firmemente el trazo izquierdo, o con menor presión de tinta, los píxeles activos coinciden con los del 3.
- **9 y 4 (o 9 y 8):** El bucle superior cerrado del 9 y la curvatura del 8 ocupan las mismas celdas de la cuadrícula superior; igualmente, un 4 escrito de forma rápida o cerrada en la parte alta presenta una silueta idéntica a un 9 inclinado.

---

### **Pregunta 2.** ¿Qué modelo tuvo menos errores? ¿Alguno tiene una exactitud de entrenamiento mucho mayor que la de prueba? ¿Qué significa eso?

- **Modelo con menos errores:**  
  El modelo **SVM** fue el más eficiente, cometiendo únicamente **4 errores** en las 450 imágenes de prueba, lo que equivale a una exactitud de prueba del **99.1% (0.991)**. Superó a KNN (7 errores, 98.4%) y a la Red Neuronal (17 errores, 96.2%).

- **Diferencia entre entrenamiento y prueba:**  
  Sí, la **Red Neuronal (MLP)** presentó una disparidad notable: alcanzó una exactitud perfecta del **100.0% (1.000)** en el conjunto de entrenamiento (0 errores), pero cayó al **96.2% (0.962)** en el conjunto de prueba con 17 fallos.

- **¿Qué significa eso? (Sobreajuste / Overfitting):**  
  Significa que la red neuronal sufrió de **sobreajuste (overfitting)**. Al disponer de una capa oculta de 64 neuronas densamente conectadas y 500 épocas de optimización, el modelo terminó "memorizando" el ruido, pequeñas imperfecciones y particularidades caligráficas exclusivas del conjunto de entrenamiento. En lugar de extraer las reglas geométricas universales de cada número, se especializó en exceso en los datos de práctica y perdió capacidad de generalización al enfrentarse a imágenes nuevas en el conjunto de prueba.

---

### **Pregunta 3.** En la matriz de confusión, ¿qué números se confunden? ¿Por qué crees que pasa?

En la matriz de confusión del modelo SVM se observaron 4 errores puntuales:
1. **El dígito 8 clasificado erróneamente como 1 (2 casos):**  
   Corresponde a los índices 25 y 255 del conjunto de prueba. Ocurre cuando el 8 fue trazado con bucles laterales muy angostos o con poca intensidad en los bordes, de modo que al promediar en bloques de 8×8 píxeles solo resalta la columna vertical densa del centro, simulando un 1.
2. **El dígito 9 clasificado erróneamente como 7 (1 caso):**  
   Corresponde al índice 240. Ocurre cuando la curvatura izquierda del bucle del 9 es tenue o incompleta; el modelo detecta la barra superior y la diagonal inferior, deduciendo que se trata de un 7.
3. **El dígito 5 clasificado erróneamente como 9 (1 caso):**  
   Corresponde al índice 242. Ocurre cuando el trazo medio del 5 se curva o une hacia arriba, asemejando un óvalo cerrado en la parte superior.

**Causa fundamental:**  
La resolución de 8×8 píxeles (64 valores) es extremadamente reducida. A esta escala se pierde la información de alta frecuencia: no hay bordes afilados ni separación clara entre trazos cercanos. Por ende, cualquier ligera desviación en la caligrafía humana genera un patrón numérico que cruza la frontera de decisión hacia otro dígito.

---

### **Pregunta 4.** Dibuja dos números propios: uno que los modelos reconozcan y otro que los confunda. ¿Qué cambiaste para confundirlos?

#### **1. Número claramente reconocible (Dígito 7):**
```python
numero_reconocible = np.array([
    [0, 8, 16, 16, 16, 16,  8, 0],
    [0, 4,  8,  8,  8, 16, 10, 0],
    [0, 0,  0,  0,  6, 16,  4, 0],
    [0, 0,  0,  0, 14, 10,  0, 0],
    [0, 0,  0,  6, 16,  2,  0, 0],
    [0, 0,  0, 14, 10,  0,  0, 0],
    [0, 0,  4, 16,  4,  0,  0, 0],
    [0, 0,  8, 12,  0,  0,  0, 0],
])
```
- **Resultado:**
  - KNN (k=3): **7**
  - SVM: **7**
  - Red Neuronal: **7**
  - *Todos los modelos coincidieron con absoluta certeza.*

#### **2. Número confuso (Híbrido ambiguo 1 / 3 / 7):**
```python
numero_confuso = np.array([
    [0, 0, 10, 15, 14,  0,  0, 0],
    [0, 0,  0,  2, 14,  0,  0, 0],
    [0, 0,  0,  2, 14,  0,  0, 0],
    [0, 0,  0,  4, 14,  0,  0, 0],
    [0, 0,  0,  4, 14,  0,  0, 0],
    [0, 0,  0,  2, 14,  0,  0, 0],
    [0, 0,  0,  2, 14,  0,  0, 0],
    [0, 0,  0,  2, 14,  0,  0, 0],
])
```
- **Resultado:**
  - KNN (k=3): **1**
  - SVM: **1**
  - Red Neuronal: **3**
  - *Hubo discrepancia entre los modelos: KNN y SVM votaron por el 1, mientras que la Red Neuronal lo clasificó como un 3.*

#### **3. ¿Qué se cambió para confundirlos?:**
Se tomó la columna vertical continua característica del 1, pero se le agregó una masa horizontal asimétrica de píxeles oscuros (intensidades 10, 15, 14) en la parte superior izquierda de la fila 0. Esto rompió la simetría del 1 sin llegar a formar la barra completa de un 7, activando los pesos intermedios de la Red Neuronal asociados a curvas de 3, mientras que los algoritmos de distancia espacial (KNN y SVM) se mantuvieron anclados al cuerpo vertical.

---

## 5. Conclusiones *(En exactamente 5 líneas y con 2+ datos numéricos)*

Para clasificar códigos postales seleccionaría el modelo **SVM** debido a su confiabilidad y mínima tasa de fallos.  
En el conjunto de prueba alcanzó una exactitud del **99.1%**, cometiendo únicamente **4 errores** de 450 imágenes.  
Superó a la Red Neuronal, la cual sufrió sobreajuste severo cayendo al 96.2% de exactitud con 17 equivocaciones.  
A diferencia de KNN, una vez entrenado, SVM no requiere comparar cada imagen contra las 1,347 muestras de entrenamiento en disco.  
Por tanto, SVM garantiza un procesamiento postal veloz en tiempo real y reduce drásticamente las cartas desviadas.  

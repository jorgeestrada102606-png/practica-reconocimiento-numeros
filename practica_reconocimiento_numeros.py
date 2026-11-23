"""
Práctica: Reconocimiento de números escritos a mano
Dataset: load_digits de scikit-learn (1,797 imágenes 8x8)
Modelos: KNN, SVM y Red Neuronal (MLP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def main():
    print("==================================================")
    print(" PASO 2: Cargar los datos")
    print("==================================================")
    digitos = load_digits()
    X = digitos.data   # las imágenes convertidas en números (64 características)
    y = digitos.target # el número que representa cada imagen (0 a 9)
    print("Imágenes:", X.shape[0])
    print("Números por imagen (píxeles):", X.shape[1])

    print("\n==================================================")
    print(" PASO 3: Ver las imágenes y contar distribución")
    print("==================================================")
    conteos = pd.Series(y).value_counts().sort_index()
    print("Imágenes de cada número:")
    print(conteos)

    # Gráfico de las primeras 20 imágenes
    fig, ejes = plt.subplots(2, 10, figsize=(12, 3))
    for i, eje in enumerate(ejes.ravel()):
        eje.imshow(digitos.images[i], cmap="gray_r")
        eje.set_title(str(y[i]))
        eje.axis("off")
    plt.suptitle("Primeras 20 imágenes del dataset")
    plt.tight_layout()
    plt.savefig("paso3_primeras_20_imagenes.png")
    plt.close()
    print("Gráfico guardado: paso3_primeras_20_imagenes.png")

    print("\n==================================================")
    print(" PASO 4: Una imagen es una tabla de números")
    print("==================================================")
    print("El número de la primera imagen es:", y[0])
    print(digitos.images[0].astype(int))

    plt.figure(figsize=(4, 4))
    plt.imshow(digitos.images[0], cmap="gray_r")
    for fila in range(8):
        for col in range(8):
            valor = int(digitos.images[0][fila, col])
            plt.text(col, fila, valor, ha="center", va="center",
                     color="white" if valor > 8 else "black")
    plt.axis("off")
    plt.title(f"Imagen 0 con valores de píxeles (Dígito: {y[0]})")
    plt.tight_layout()
    plt.savefig("paso4_matriz_pixeles_0.png")
    plt.close()
    print("Gráfico guardado: paso4_matriz_pixeles_0.png")

    print("\n==================================================")
    print(" PASO 5: Separar entrenamiento y prueba")
    print("==================================================")
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print("Entrenamiento:", len(X_ent))
    print("Prueba:", len(X_prueba))

    print("\n==================================================")
    print(" PASO 6: La línea base (DummyClassifier)")
    print("==================================================")
    tonto = DummyClassifier(strategy="most_frequent").fit(X_ent, y_ent)
    acc_dummy = round(tonto.score(X_prueba, y_prueba), 3)
    print("Exactitud de la línea base:", acc_dummy)

    print("\n==================================================")
    print(" PASO 7: Entrenar tres modelos")
    print("==================================================")
    modelos = {
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(),
        "Red neuronal": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X_ent, y_ent)
        print("Entrenado:", nombre)

    print("\n==================================================")
    print(" PASO 8: Evaluar con el conjunto de prueba")
    print("==================================================")
    filas = []
    for nombre, modelo in modelos.items():
        p = modelo.predict(X_prueba)
        filas.append({
            "Modelo": nombre,
            "Exactitud entrenamiento": modelo.score(X_ent, y_ent),
            "Exactitud prueba": accuracy_score(y_prueba, p),
            "Errores": int((p != y_prueba).sum())
        })
    df_eval = pd.DataFrame(filas).round(3)
    print(df_eval.to_string(index=False))

    # Matriz de confusión para SVM
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(modelos["SVM"], X_prueba, y_prueba, cmap="Blues", ax=ax)
    plt.title("SVM: filas = número real, columnas = predicción")
    plt.tight_layout()
    plt.savefig("paso8_matriz_confusion_svm.png")
    plt.close()
    print("Matriz de confusión guardada: paso8_matriz_confusion_svm.png")

    print("\n==================================================")
    print(" PASO 9: Ver los errores del modelo SVM")
    print("==================================================")
    p_svm = modelos["SVM"].predict(X_prueba)
    errores = np.where(p_svm != y_prueba)[0]
    print(f"Total de errores en SVM: {len(errores)}")
    for idx in errores:
        print(f" - Índice {idx}: Real={y_prueba[idx]}, Predicho={p_svm[idx]}")

    if len(errores) > 0:
        fig, ejes = plt.subplots(1, len(errores), figsize=(2.2 * len(errores), 2.5))
        for eje, i in zip(np.atleast_1d(ejes), errores):
            eje.imshow(X_prueba[i].reshape(8, 8), cmap="gray_r")
            eje.set_title(f"Real: {y_prueba[i]}\nDijo: {p_svm[i]}")
            eje.axis("off")
        plt.suptitle("Errores cometidos por SVM en conjunto de prueba")
        plt.tight_layout()
        plt.savefig("paso9_errores_svm.png")
        plt.close()
        print("Gráfico de errores guardado: paso9_errores_svm.png")

    print("\n==================================================")
    print(" PASO 10: Dibuja tu propio número")
    print("==================================================")
    # Ejemplo 1: Dígito 7 reconocible (el del enunciado)
    digito_reconocible = np.array([
        [0, 8, 16, 16, 16, 16, 8, 0],
        [0, 4, 8, 8, 8, 16, 10, 0],
        [0, 0, 0, 0, 6, 16, 4, 0],
        [0, 0, 0, 0, 14, 10, 0, 0],
        [0, 0, 0, 6, 16, 2, 0, 0],
        [0, 0, 0, 14, 10, 0, 0, 0],
        [0, 0, 4, 16, 4, 0, 0, 0],
        [0, 0, 8, 12, 0, 0, 0, 0],
    ])

    # Ejemplo 2: Dígito ambiguo/confuso (híbrido entre 3 y 8)
    digito_confuso = np.array([
        [0, 6, 14, 14, 14, 12, 0, 0],
        [0, 8, 0, 0, 0, 14, 2, 0],
        [0, 6, 0, 0, 4, 14, 0, 0],
        [0, 0, 4, 12, 14, 8, 0, 0],
        [0, 4, 0, 0, 0, 14, 0, 0],
        [0, 8, 0, 0, 0, 14, 2, 0],
        [0, 6, 12, 14, 14, 10, 0, 0],
        [0, 0, 2, 4, 4, 2, 0, 0],
    ])

    print("\nPredicciones para Dígito Reconocible (7 claro):")
    for nombre, modelo in modelos.items():
        pred = modelo.predict([digito_reconocible.ravel()])[0]
        print(f" - {nombre} predice: {pred}")

    print("\nPredicciones para Dígito Confuso (diseñado ambiguo 3/8/5):")
    for nombre, modelo in modelos.items():
        pred = modelo.predict([digito_confuso.ravel()])[0]
        print(f" - {nombre} predice: {pred}")

    # Guardar ambos dígitos dibujados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))
    ax1.imshow(digito_reconocible, cmap="gray_r")
    ax1.set_title("Número Reconocible (7)")
    ax1.axis("off")
    ax2.imshow(digito_confuso, cmap="gray_r")
    ax2.set_title("Número Confuso (3/8)")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig("paso10_numeros_propios.png")
    plt.close()
    print("Gráfico de números propios guardado: paso10_numeros_propios.png")

if __name__ == "__main__":
    main()

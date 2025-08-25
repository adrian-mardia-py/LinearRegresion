import numpy as np
from LinealRegresion import regresionLineal

from time import time

def main() -> None:
    """
    Pequeña muestra simple de las capacidades del modelo de regresión lineal
    """
    dispersion = float(input("Introduzca como de difuminado quiere la muestra-> "))
    m = int(input("Introduzca la pendiente -> "))
    n = int(input("Iltroduzca la ordenada en el origen -> "))
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = m * X + n + np.random.randn(100, 1) * dispersion

    iteraciones = int(input("\nIntroduzca el numero de iteraciones que desea realizar. \nPreferiblemente entre 5000 y 10000\n\n-------> "))

    modelo1 = regresionLineal(target = 'gpu', n_iter = iteraciones, printECM=False)
    modelo2 = regresionLineal(target = 'cpu', n_iter = iteraciones, printECM=False)

    a = time()
    modelo2.fit_transform(X, y, plot=False)
    b = time()

    c = time()
    modelo1.fit_transform(X, y, plot=False)
    d = time()

    print(f"\nEl modelo usando la CPU ha tardado {b-a} segundos mientras que en la GPU se ha tardado {d-c} segundos\n")

    #Mostramos la gráfica
    modelo1.fit_transform(X, y, plot=True)



if __name__ == "__main__":
    main()
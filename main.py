import numpy as np
from LinealRegresion import regresionLineal

def main() -> None:
    """
    Pequeña muestra simple de las capacidades del modelo de regresión lineal
    """
    dispersion = float(input("Introduzca como de difuminado quiere la muestra-> "))
    m = int(input("Introduzca la pendiente -> "))
    n = int(input("Iltroduzca la ordenada en el origen -> "))
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = m * X + n + np.random.randn(100, 1) * dispersion

    iteraciones = int(input("\nIntroduzca el numero de iteraciones que desea realizar. \nDeben ser más de 300 iteraciones, si no se fijarán automáticamente\n\n-------> "))
    gpu = bool(input("\n¿Quiere hacer los cálculos en GPU? (Si/No) --> ").upper() == 'SI')

    modelo = regresionLineal(target = 'gpu' if gpu else 'cpu', n_iter = iteraciones, printECM=True)

    modelo.fit_transform(X, y, plot=True)


if __name__ == "__main__":
    main()
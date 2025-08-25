"""
Creado el 25/08/2025 por Adrián Martín Díaz

Este programa trata de implementar una clase de regresión lineal al estilo de sklearn.linear_model.LinearRegresion.
Se puede importar en otro script para usarse o si se ejecuta este programa hace una pequeña demo con parámetros introducidos por el usuario
"""

import numpy as np
import matplotlib.pyplot as plt

from numba import cuda

import warnings
from numba.core.errors import NumbaPerformanceWarning

# Ignorar solo warnings de NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


class regresionLineal():
    def __init__(
            self, target: str = 'cpu', 
            n_iter: np.int32 = 1000, 
            alpha: np.float32 = 0.001,
            printECM: bool = True
    ) -> None: 
        """
        Parámetros:
        -----------
        target: str 'cpu'/'gpu' -> permite diferenciar entre si hacer las operaciones en la cpu o en la gpu
        n_iter: np.int32=1000 -> Permite determinar el número de iteraciones que se quieren llevar a cabo. Siempre mas de 100
        alpha: np.float32=0.001 -> Permite determinar el parámetro de aprendizaje con el que se empieza la regresión
        printECM: bool=True -> Permite ver como baja el error cuadrático medio a lo largo de las iteraciones 
        """
        
        #Definimos parámetros que condicionarán a la clase (hiperparametros)
        assert (target == 'cpu') or (target == 'gpu'), 'Solo se pueden realizar los cálculos en cpu/gpu'
        self.target: str = target

        assert (n_iter is int) or (n_iter > 100), 'Las iteraciones deben ser un entero positivo mayor de 100'
        self.n_iter = n_iter

        assert (alpha is float) or (alpha > 0), 'El parámetro de aprendizaje debe ser un numero positivo'
        self.alpha: np.float32 = alpha

        self.print = printECM

        #Definimos algunos parámetros importantes para el ajuste lineal
        self.n_parametros: np.int32 = 0
        self.n_puntos: np.int32 = 0
        self.error_cuadratico: np.float32 = 0.0
        self.error_provisional = np.float32('inf')

        #Definimos algunos arrays importantes para el ajuste lineal, todos vectores columna
        self.parametros = np.ones((2, 1))
        self.gradiente = np.zeros_like(self.parametros)
        self.h = np.zeros_like(self.parametros)
        self.matriz_puntos = np.zeros_like(self.parametros)


    def definirVariables(self, X: np.ndarray) -> None:
        """
        Esta función obtendrá a partir de la matriz de puntos:
        -La matriz de los puntos con una fila de unos inicial
        -el numero de puntos
        -El numero de parámetros
        -Definirá los tamaños del gradiente, la matriz de parámetros y la función h

        Parámetros:
        -----------
        X: np.ndarray -> La matriz de puntos

        Devuelve:
        ---------
        None
        """
        bias = np.ones_like(X[:, 0]).reshape(-1, 1)
        self.matriz_puntos = np.hstack((bias, X), dtype=np.float32)

        #Definimos el numero de puntos y de parámetros
        self.n_puntos, self.n_parametros = self.matriz_puntos.shape

        #Modelamos los demás arrays en función de esos parametros
        self.parametros = np.ones((self.n_parametros, 1))
        self.gradiente = np.ones_like(self.parametros)
        self.h = np.zeros((self.n_puntos, 1))


    def __gradienteCPU(self, y: np.ndarray) -> None:
        """
        Calcula el descenso en gradiente a través de la CPU

        Parámetros:
        ----------
        y: np.ndarray -> La matriz solución

        Devuelve:
        ---------
        None
        """
        for i in range(self.n_iter):
            #Ponemos el gradiente y el error cuadratico a 0
            self.gradiente = np.zeros_like(self.parametros)
            self.error_cuadratico: np.float32 = 0.0 

            for j in range(self.n_puntos):
                #Calculamos tanto el gradiente como el error cuadratico
                self.gradiente = self.gradiente + (np.matmul(self.matriz_puntos[j, :], self.parametros) - y[j, 0]) * self.matriz_puntos[j, :].reshape(-1, 1)
                self.error_cuadratico = self.error_cuadratico + 0.5 * (np.matmul(self.matriz_puntos[j, :], self.parametros) - y[j, 0])[0]**2 / self.n_puntos

            #Si a mitad de las iteraciones, o luego de 1000 el error empieza a aumetar, significa q estamos "rebotando" y salimos del ciclo
            if self.error_cuadratico > self.error_provisional:
                self.alpha = self.alpha*0.1

                if (i>(self.n_iter/2) or i>1000) and self.alpha <= 1 * 10**-10:
                    print(f'El error cuadratico en el que se ha roto es {self.error_cuadratico}, en la iteración {i}')
                    break
            
            self.error_provisional = self.error_cuadratico

            if self.print:
                print("ECM-> ", self.error_cuadratico)

            #Ajustamos los parámetros
            self.parametros = self.parametros - self.alpha * self.gradiente

        print("La matriz de parámetros es -> ", *self.parametros)


    def __gradienteGPU(self, y) -> None:
        """
        Calcula el descenso en gradiente a través de la GPU

        Parámetros:
        ----------
        y: np.ndarray -> La matriz solución

        Devuelve:
        ---------
        None
        """
        alpha = np.array([self.alpha], dtype=np.float32)

        #Definimos los hilos por bloque y por malla
        hilos = 64
        bloques_por_malla = (self.n_puntos + hilos - 1) // hilos 

        #Alocamos las variables en memoria
        matriz_puntos_cuda = cuda.to_device(self.matriz_puntos)
        y_cuda = cuda.to_device(y)
        parametros_cuda = cuda.to_device(self.parametros)
        gradiente_cuda = cuda.to_device(self.gradiente)
        error_cuadratico_matriz = np.array([self.error_cuadratico], dtype=np.float32)
        error_cuadratico_cuda = cuda.to_device(error_cuadratico_matriz)
        h_cuda = cuda.to_device(self.h)
        alpha_cuda = cuda.to_device(alpha)

        #Lanzamos el kernel dentro de un bucle
        for _ in range(self.n_iter):
            #Ponemos a 0 tanto el error cuadrático como el gradiente
            gradiente_cuda.copy_to_device(np.zeros_like(self.gradiente))
            error_cuadratico_cuda.copy_to_device(np.zeros_like(error_cuadratico_matriz))
            h_cuda.copy_to_device(np.zeros_like(self.h))


            #Lanzamos el kernel
            descensoGradiente[bloques_por_malla, hilos](
                self.n_puntos,
                self.n_parametros,
                h_cuda,
                parametros_cuda,
                matriz_puntos_cuda,
                gradiente_cuda,
                y_cuda,
                error_cuadratico_cuda
            )
            cuda.synchronize()

            #Actualizamos los parámetros
            actualizarParametros[1, self.n_parametros](
                parametros_cuda,
                gradiente_cuda,
                alpha_cuda,
                self.n_parametros,
                self.n_puntos
            )
            cuda.synchronize()

            #Actualizamos alpha TODO: Comprobar que funcione
            if self.actualizarAlphaGpu(
                error_cuadratico_cuda,
                error_cuadratico_matriz,
                _,
                alpha_cuda,
                alpha
            ):
                print("----------------------------------------------")
                break

            if self.print:
                print("ECM-> ", error_cuadratico_matriz[0])

        #Obtenemos la matriz de parámetros calculada
        parametros_cuda.copy_to_host(self.parametros)
        print("La matriz de parámetros es -> ", *self.parametros)


    def actualizarAlphaGpu(
            self,
            error_cuadratico_cuda,
            error_cuadratico_matriz,
            _,
            alpha_cuda,
            alpha
    ):
        """
        Actualiza alpha haciéndolo más pequeño si rebotamos en la solución

        Parámetros:
        -----------
        error_cuadratico_cuda: DeviceNDArray -> Error cuadrático medio en la GPU.
        error_cuadratico_matriz: np.ndarray -> Array en CPU para copiar el error desde la GPU.
        _: int -> Iteración actual.
        alpha_cuda: DeviceNDArray -> Valor de alpha en GPU.
        alpha: np.ndarray -> Copia de alpha en CPU para modificarlo.

        Devuelve:
        ---------
        bool | None -> True si se cumple la condición de rebote, None en caso contrario.
        """
        error_cuadratico_cuda.copy_to_host(error_cuadratico_matriz)
        alpha_cuda.copy_to_host(alpha)

        if error_cuadratico_matriz[0] > self.error_provisional:
            #Si ha 'rebotado', entonces disminuimos el parámetro de aprendizaje
            alpha = alpha * 0.1
            print("Se ha corregido alpha a -----------------------> ", alpha)

            if (alpha[0] < 1*10**-25) and (_ > self.n_iter/2):
                print(f"Se ha roto en la iteración {_} con un error de {error_cuadratico_matriz}, y un alpha de {alpha[0]}")
                return True
            
        self.error_provisional = error_cuadratico_matriz[0]


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo de regresión lineal a partir de los datos de entrenamiento.

        Parámetros:
        -----------
        X: np.ndarray -> Matriz de puntos de tamaño.
        y: np.ndarray -> Vector objetivo de tamaño.

        Devuelve:
        ---------
        None
        """
        #Definimos las variables del problema
        self.definirVariables(X)

        if self.target == 'cpu':
            self.__gradienteCPU(y)
        else:
            self.__gradienteGPU(y)


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Genera las predicciones del modelo a partir de un conjunto de datos.

        Parámetros:
        -----------
        X: np.ndarray -> Matriz de puntos.

        Devuelve:
        ---------
        y: np.ndarray -> Vector de predicciones.
        """
        #Ajustamos el término intependiente
        bias: np.ndarray = np.ones_like(X[:, 0]).reshape(-1, 1)
        self.matriz_puntos = np.hstack((bias, X))

        y = np.zeros_like(X[:, 0])

        #Aplicamos la fórmula de la regresion lineal
        for i in range(self.matriz_puntos.shape[0]):
            y[i] = np.matmul(self.matriz_puntos[i, :], self.parametros)[0] #Para no pasar automaticamente de un array a un numero

        return y
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, plot: bool = False):
        """
        Entrena el modelo con los datos de entrada y devuelve las predicciones.

        Opcionalmente representa gráficamente la regresión lineal.

        Parámetros:
        -----------
        X: np.ndarray -> Matriz de puntos.
        y: np.ndarray -> Vector solución.
        plot: bool=False -> Si es True, muestra la gráfica del ajuste.

        Devuelve:
        ---------
        ajuste: np.ndarray -> Vector de predicciones del modelo sobre X.
        """
        self.fit(X, y),
        ajuste = self.transform(X)

        if not plot:
            return ajuste
        else:
            plt.scatter(X, y, color='dodgerblue', label='Datos', edgecolors='white', s=400, alpha=0.15)
            plt.plot(X, ajuste, color='darkorange', linewidth=2)
            plt.title("Regresión lineal por descenso en gradiente")

            plt.show()


@cuda.jit
def descensoGradiente(
    n_puntos,
    n_parametros,
    h_cuda,
    parametros_cuda,
    matriz_puntos_cuda,
    gradiente_cuda,
    y_cuda,
    error_cuadratico_cuda
) -> None:
    """
    Kernel CUDA que calcula el descenso en gradiente para regresión lineal.

    Parámetros:
    -----------
    x: DeviceNDArray -> Matriz de puntos.
    y: DeviceNDArray -> Vector solución.
    parametros: DeviceNDArray -> Parámetros del modelo.
    alpha: DeviceNDArray -> Tasa de aprendizaje.
    error_cuadratico: DeviceNDArray -> Acumulador del error cuadrático medio.
    
    Devuelve:
    ---------
    None
    """

    idx = cuda.grid(1)

    if idx < n_puntos:
        
        #Calculamos la funcion h
        for kdx in range(n_parametros):
            cuda.atomic.add(h_cuda, (idx, 0), parametros_cuda[kdx, 0] * matriz_puntos_cuda[idx, kdx])

        #Calculamos el error cuadrático
        cuda.atomic.add(error_cuadratico_cuda, 0, ((h_cuda[idx, 0] - y_cuda[idx, 0])**2)/n_puntos)

        #calculamos el gradiente
        for kdx in range(n_parametros):
            cuda.atomic.add(gradiente_cuda, (kdx, 0), (h_cuda[idx, 0] - y_cuda[idx, 0]) * matriz_puntos_cuda[idx, kdx])
        

@cuda.jit
def actualizarParametros(
    parametros_cuda,
    gradiente_cuda,
    alpha,
    n_parametros,
    n_puntos
) -> None:
    """
    Kernel CUDA que actualiza los parámetros del modelo en cada iteración.

    Parámetros:
    -----------
    parametros: DeviceNDArray -> Parámetros del modelo.
    gradientes: DeviceNDArray -> Gradientes calculados en la iteración.
    alpha: DeviceNDArray -> Tasa de aprendizaje.

    Devuelve:
    ---------
    None
    """
    
    idx= cuda.grid(1)

    #Actualizamos los parámetros
    if idx < n_parametros:
        cuda.atomic.sub(parametros_cuda, (idx, 0), alpha[0] * gradiente_cuda[idx, 0] / n_puntos)

"""
Segundo Parcial Modelado Continuo
-- Series de Fouirer --

5 de junio de 2024

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, rfft, fftfreq, fftshift, ifftshift
#%%

# Ejercicio 1 

#(c)

def regla_trapecio(f_vals, a, b):
    n = len(f_vals) - 1
    h = (b - a) / n
    return h * (0.5 * f_vals[0] + np.sum(f_vals[1:-1] + 0.5 * f_vals[-1]))

def coeficientes_fourier(f, N, num_puntos=1000):
    a = -np.pi
    b = np.pi
    x = np.linspace(a, b, num_puntos)
    fx = f(x)

    a0 = (1 / np.pi) * regla_trapecio(fx, a, b)
    an = []
    bn = []

    for n in range(1, N + 1):
        an_val = (1 / np.pi) * regla_trapecio(fx * np.cos(n * x), a, b)
        bn_val = (1 / np.pi) * regla_trapecio(fx * np.sin(n * x), a, b)
        an.append(an_val)
        bn.append(bn_val)

    def serie_truncada(x_eval):
        x_eval = np.array(x_eval)
        resultado = a0 / 2
        for n in range(1, N + 1):
            resultado += an[n - 1] * np.cos(n * x_eval) + bn[n - 1] * np.sin(n * x_eval)
        return resultado

    return serie_truncada, a0, an, bn

def f(x):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    return np.piecewise(
        x,
        [x < -np.pi/3, (-np.pi/3 <= x) & (x <= np.pi/3), x > np.pi/3],
        [1, 0, -1]
    )

# Puntos y serie truncada
x_vals = np.linspace(-np.pi, np.pi, 1024)
N = 20  # Número de términos
serie, a0, an, bn = coeficientes_fourier(f, N)

# Gráfica
plt.plot(x_vals, f(x_vals), label="f(x)")
plt.plot(x_vals, serie(x_vals), label=f"Serie de Fourier truncada (N={N})")
plt.xlabel("x")
plt.grid()
plt.legend()
plt.title("Aproximación de f(x) por la serie de Fourier truncada")
plt.show()

# (d)

# Ya tienes esta función definida arriba:
# regla_trapecio, coeficientes_fourier, f

def Sn(f, N):
    _, a0, an, bn = coeficientes_fourier(f, N)
    pepe = a0**2
    suma = 0
    for k in range(N):
        suma += an[k]**2 + bn[k]**2
    return pepe + (1/2)*suma

# Calcular la energía real de la función: ||f||^2 = (1/π) ∫ f^2(x) dx
def energia_real(f):
    x_vals = np.linspace(-np.pi, np.pi, 1024)
    fx_squared = f(x_vals)**2
    return (1/np.pi) * regla_trapecio(fx_squared, -np.pi, np.pi)

# Función para calcular el error relativo |(S_N - ||f||²)| / ||f||²
def error_parseval(Sn_values, energia_real):
    return np.abs(Sn_values - energia_real)

# Valores de N (usaremos más puntos para mejor visualización)
N_values = np.arange(1, 201, 5)  # N desde 1 hasta 200, en pasos de 5
Sn_values = [Sn(f, N) for N in N_values]
energia = energia_real(f)

# Calculamos el error relativo
error = error_parseval(Sn_values, energia)

# Graficamos la convergencia del error (en escala logarítmica)
plt.figure(figsize=(12, 6))

# Subgráfico 1: Convergencia de S_N
plt.subplot(1, 2, 1)
plt.plot(N_values, Sn_values, 'b-', marker='o', markersize=4, label=r"$S_N$ (Parseval truncado)")
plt.axhline(energia, color='r', linestyle='--', label=r"$\|f\|^2$ exacta")
plt.xlabel("N (términos de la serie)")
plt.ylabel("Energía")
plt.title("Convergencia de $S_N$ a $\|f\|^2$")
plt.grid(True)
plt.legend()

# Subgráfico 2: Tasa de error (escala log-log)
plt.subplot(1, 2, 2)
plt.loglog(N_values, error, 'k-', marker='s', markersize=4, label="Error relativo")
plt.xlabel("N (escala log)")
plt.ylabel("Error relativo (escala log)")
plt.title("Tasa de convergencia del error")
plt.grid(True, which="both", linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()

#%%

# Ejercicio 2

# (b)

# Parámetros
Delta_x = 0.1
f0 = 5  # Hz, debe ser < 1/(2 * Delta_x) = 5
x_k = np.arange(-10, 11) * Delta_x
f_k = np.sinc(2 * np.pi * f0  * x_k)

# Función sinc normalizada
def sinc(x):
    return np.sinc(x)  # np.sinc(x) = sin(pi x)/(pi x)

# Puntos finos para graficar interpolación
x = np.linspace(-2, 2, 1000)
f_interp = np.zeros_like(x)

# Fórmula de Shannon
for k, fk in zip(x_k, f_k):
    f_interp += fk * sinc((x - k) / Delta_x)

# Gráfica
plt.figure(figsize=(10, 5))
plt.plot(x, f_interp, label='Interpolación Shannon', color='blue')
plt.stem(x_k, f_k, linefmt='gray', markerfmt='ro', basefmt=" ", label='Muestras $f_k$', use_line_collection=True)
plt.title('Interpolación de Shannon usando función sinc')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


# (c)

# Número de subintervalos
N = 50
x_k = np.linspace(-3, 3, N + 1)  # Grilla regular

# Funciones originales evaluadas en la grilla
f_k = np.where(np.abs(x_k) <= 1, 1, 0)        # f(x) = chi_{[-1,1]}(x)
g_k = np.exp(-x_k**2)                         # g(x) = exp(-x^2)

# Delta_x para la fórmula de Shannon
Delta_x = x_k[1] - x_k[0]

# Dominio fino para graficar la reconstrucción
x = np.linspace(-3, 3, 1000)

# Función sinc normalizada
def sinc(x):
    return np.sinc(x)  # Normalizada como sin(pi x)/(pi x)

# Interpolación de Shannon
def shannon_interpolation(x, x_k, f_k, Delta_x):
    f_interp = np.zeros_like(x)
    for k, fk in zip(x_k, f_k):
        f_interp += fk * sinc((x - k) / Delta_x)
    return f_interp

# Interpolaciones
f_interp = shannon_interpolation(x, x_k, f_k, Delta_x)
g_interp = shannon_interpolation(x, x_k, g_k, Delta_x)

# Graficar ambas funciones y sus aproximaciones
plt.figure(figsize=(12, 6))

# Gráfica de f(x)
plt.subplot(1, 2, 1)
plt.plot(x, f_interp, label='Interpolación Shannon de f', color='blue')
plt.stem(x_k, f_k, linefmt='gray', markerfmt='ro', basefmt=" ", use_line_collection=True, label='Muestras de f')
plt.title('Función indicadora $\\chi_{[-1,1]}(x)$')
plt.xlabel('x')
plt.grid(True)
plt.legend()

# Gráfica de g(x)
plt.subplot(1, 2, 2)
plt.plot(x, g_interp, label='Interpolación Shannon de g', color='green')
plt.stem(x_k, g_k, linefmt='gray', markerfmt='ro', basefmt=" ", use_line_collection=True, label='Muestras de g')
plt.title('Función gaussiana $e^{-x^2}$')
plt.xlabel('x')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()










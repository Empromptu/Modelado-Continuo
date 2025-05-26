# Ejercicios Guia 2 Modelado Continuo

import numpy as np
import matplotlib.pyplot as plt

#%%
import pyaudio

# Frecuencias del acorde de LA mayor
frecuencias = [277.18, 329.63, 440.00]  # C♯4, E4, A4

# Parámetros de la señal
duracion = 1.0  # segundos
fs = 44100  # frecuencia de muestreo
t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)

# Generar señal compuesta
senal = sum(np.sin(2 * np.pi * f * t) for f in frecuencias)
senal /= len(frecuencias)  # normalizar

# Convertir a bytes (16-bit PCM)
senal_int16 = np.int16(senal * 32767)
senal_bytes = senal_int16.tobytes()

# Reproducir con PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fs,
                output=True)

print("Reproduciendo...")
stream.write(senal_bytes)
stream.stop_stream()
stream.close()
p.terminate()

# Graficar 2-3 periodos de la frecuencia más baja (~277 Hz)
periodo = 1 / min(frecuencias)
muestras_por_periodo = int(fs * periodo)
plt.plot(t[:3 * muestras_por_periodo], senal[:3 * muestras_por_periodo])
plt.title("Señal compuesta (C♯4 + E4 + A4)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()


# GRAFICO LA TRANSFORMADA :
    
# Transformada de Fourier
F = np.fft.fft(senal)
frecuencias_fft = np.fft.fftfreq(len(senal), 1/fs)

# Solo nos interesa la parte positiva (simétrica)
positivas = frecuencias_fft > 0
frecuencias_pos = frecuencias_fft[positivas]
F_pos = np.abs(F[positivas])

# Graficar la magnitud de la transformada
plt.figure(figsize=(10, 5))
plt.plot(frecuencias_pos, F_pos)
plt.title("Transformada de Fourier de la señal compuesta")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 1000)  # mostrar solo hasta 1000 Hz para claridad
plt.grid(True)
plt.show()

N = len(senal)
frecuencia_muestreo = 44100  # por ejemplo

frecuencias = np.fft.fftfreq(N, d=1/frecuencia_muestreo)
transformada = np.fft.fft(senal)

# Centrar ambos
frecuencias_shift = np.fft.fftshift(frecuencias)
transformada_shift = np.fft.fftshift(np.abs(transformada))

# Graficar
plt.plot(frecuencias_shift, transformada_shift)
plt.title("FFT centrada")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(-1000, 1000)    # Zoom al rango de frecuencias de -1000 Hz a 1000 Hz
plt.ylim(0, 7000)         # Opcional: Zoom vertical si querés
plt.grid(True)
plt.show()

#%%

# Ejercicio 5

def regla_trapecio_compuesta(f,a,b,n):
    
    """
    Estima la integral definida de f en el intervalo [a, b] 
    usando la regla de los trapecios compuesta con n subintervalos.
    
    Parámetros:
    f -- función a integrar (debe ser una función que reciba un valor numérico)
    a -- límite inferior del intervalo
    b -- límite superior del intervalo
    n -- número de subintervalos (debe ser un entero positivo)
    
    Retorna:
    Aproximación de la integral de f en [a, b]
    
    """
    
    if n > 0 :
        h = np.abs(b-a) / n
        suma = 0.5 * f(a) + 0.5 * f(b)
        for i in range(1,n):
            xi = a + i*h
            suma += f(xi)
            
        return h * suma

# Ejercicio 6

def f(x) :
    return np.exp(-x**2)

a = -5
b = 5
n = 10000

print((regla_trapecio_compuesta(f, a, b, n),np.sqrt(np.pi)))

#%%

# Ejercicio 7

# (a)

def Dirichlet(t,N):
    numerador = np.sin((N+1/2)*t)
    denominador = 2 * np.sin(t/2)
    if denominador.any() == 0 :
        return "Dividiendo por 0, np.sin(t/2) = 0"
    return numerador / denominador


time = np.arange(-np.pi, np.pi, 0.01)
N_values = [0, 0.5, 0.7, 1, 2]

plt.figure(figsize=(10, 6))
for n in N_values:
    plt.plot(time, Dirichlet(time, n), label=f'N = {n}')

plt.title('Funciones núcleo de Dirichlet para distintos N')
plt.xlabel('t')
plt.ylabel('D_N(t)')
plt.grid(True)
plt.legend()
plt.show()

# (b)

def regla_trapecios_compuesta_array(f_vals, a, b):
    # Acomodo la funcion de la regla de los trapecios para que soporte arrays.
    n = len(f_vals) - 1
    h = (b - a) / n
    return h * (0.5 * f_vals[0] + np.sum(f_vals[1:-1]) + 0.5 * f_vals[-1])

time = np.arange(-np.pi, np.pi, 0.01)
N_values = [0, 0.5, 0.7, 1, 2]
a = -np.pi
b = np.pi

for n in N_values:
    D_vals = Dirichlet(time,n)
    integral = regla_trapecios_compuesta_array(D_vals,a,b)
    resultado = integral / np.pi                               
    print(resultado)

#%%

# Ejercicio 8 (Fourier)

def regla_trapecio(f_vals,a,b):
    n = len(f_vals)-1
    h =(b-a)/n
    return h * (0.5 * f_vals[0] + np.sum(f_vals[1:-1] + 0.5 * f_vals[-1]))

def coeficientes_fourier(f,N,num_puntos=1000):
    a = -np.pi
    b = np.pi
    x = np.linspace(a, b, num_puntos)
    fx = f(x)
    
    a0 = (1 / np.pi) * regla_trapecio(fx, a, b)
    
    an = []
    bn = []
    
    for n in range(1, N + 1):
        cos_nx = np.cos(n * x)
        sin_nx = np.sin(n * x)
        an_val = (1/np.pi) * regla_trapecio(fx * cos_nx, a, b)
        bn_val = (1/np.pi) * regla_trapecio(fx * sin_nx, a, b)
        an.append(an_val)
        bn.append(bn_val)
        
    def serie_truncada(x_eval):
        x_eval = np.array(x_eval)
        resultado = a0 / 2
        for n in range(1, N + 1):
            resultado += an[n - 1] * np.cos(n * x_eval) + bn[n - 1] * np.sin(n * x_eval)
        return resultado

    return serie_truncada, a0, an, bn

################ ejemplo ################ 

def f(x):
    return x

N = 10
S_N, a0, an, bn = coeficientes_fourier(f, N)

# Graficar
x_vals = np.linspace(-np.pi, np.pi, 1000)
plt.plot(x_vals, f(x_vals), label="f(x) original")
plt.plot(x_vals, S_N(x_vals), label=f"Serie de Fourier truncada (N={N})")
plt.legend()
plt.grid()
plt.title("Aproximación de Fourier")
plt.show()

######################################### 
#%%

# Ejercicio 9

def g(x):
    return x**4

N = 20
S_N, a0, an, bn = coeficientes_fourier(g, N)

x_vals = np.linspace(-np.pi, np.pi,1000)
plt.plot(x_vals,g(x_vals), label ="g(x) original")
plt.plot(x_vals,S_N(x_vals), label=f"Serie de Fourier truncada (N={N})")
plt.legend()
plt.grid()
plt.title("Aproximación de Fourier")
plt.show()

#%%

# Ejercicio 10

x_vals = np.linspace(-np.pi, np.pi, 1000)

def fourier_x(x, N):
    result = np.zeros_like(x)
    for n in range(1, N + 1):
        coef = (2 * (-1)**(n+1)) / n
        result += coef * np.sin(n * x)
    return result

def fourier_x2(x, N):
    result = np.full_like(x, np.pi**2 / 3)
    for n in range(1, N + 1):
        coef = (4 * (-1)**n) / (n**2)
        result += coef * np.cos(n * x)
    return result

# Valores de N para truncar la serie
N_values = [1, 3, 5, 10, 20]

# Aproximación de f(x) = x
plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(x_vals, fourier_x(x_vals, N), label=f'N={N}')
plt.plot(x_vals, x_vals, 'k--', label='f(x) = x', linewidth=2)
plt.title('Aproximaciones de f(x) = x con Series de Fourier')
plt.legend()
plt.grid()
plt.show()

# Aproximación de f(x) = x^2
plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(x_vals, fourier_x2(x_vals, N), label=f'N={N}')
plt.plot(x_vals, x_vals**2, 'k--', label='f(x) = x²', linewidth=2)
plt.title('Aproximaciones de f(x) = x² con Series de Fourier')
plt.legend()
plt.grid()
plt.show()

#%%

# Ejercicio 11

def fenomeno_gibbs(x):
    return np.where(x <= 0, 1, 0)

N = 20
S_N, a0, an, bn = coeficientes_fourier(fenomeno_gibbs, N)

x_vals = np.linspace(-np.pi, np.pi,1000)
plt.plot(x_vals, [fenomeno_gibbs(x) for x in x_vals], label="f(x) original")
plt.plot(x_vals,S_N(x_vals), label=f"Serie de Fourier truncada (N={N})")
plt.legend()
plt.grid()
plt.title("Aproximación de Fourier")
plt.show()

#%%

# Ejercicio 12

def funcion_no_derivable(x):
    suma = 0
    N=200
    for n in range(1,N):
        seno = np.sin((n**2) * x)
        suma += seno/n**2
    return suma

N = 20
S_N, a0, an, bn = coeficientes_fourier(funcion_no_derivable, N) 

x_vals = np.linspace(-np.pi, np.pi,1000)
plt.plot(x_vals,funcion_no_derivable(x_vals), label="Funcion")
plt.plot(x_vals,S_N(x_vals),"Aproximacion")
plt.legend()
plt.grid()
plt.title("Aproximacion de Fourier")
plt.show()    

#%%

# Ejercicio 13

# Grilla gruesa: paso 1/8
x_coarse = np.arange(0, 1 + 1/8, 1/8)

f1 = np.sin(2 * np.pi * x_coarse)
f2 = np.sin(18 * np.pi * x_coarse)

plt.figure(figsize=(10, 5))
plt.scatter(x_coarse, f1, label=r'$f_1(x) = \sin(2\pi x)$', color='blue')
plt.scatter(x_coarse, f2, label=r'$f_2(x) = \sin(18\pi x)$', color='red', marker='x')
plt.title('Scatter plot sobre grilla de paso $1/8$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Grilla fina: paso 1/1000 :

x_fine = np.linspace(0, 1, 1000)
f1_fine = np.sin(2 * np.pi * x_fine)
f2_fine = np.sin(18 * np.pi * x_fine)

plt.figure(figsize=(10, 5))
plt.plot(x_fine, f1_fine, label=r'$f_1(x) = \sin(2\pi x)$', color='blue')
plt.plot(x_fine, f2_fine, label=r'$f_2(x) = \sin(18\pi x)$', color='red')
plt.scatter(x_coarse, f1, color='blue')
plt.scatter(x_coarse, f2, color='red', marker='x')
plt.title('Funciones con grilla fina y muestreo sobre grilla gruesa')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

#%%

# Ejercicio 14

from scipy.fft import fft, ifft, rfft, fftfreq, fftshift, ifftshift

def transformada_fourier_discreta(v):  # DFT
    N = len(v)
    v_k = np.zeros(N, dtype=complex)
    for k in range(N):
        suma_k = 0
        for j in range(N):
            exponente = -2j * np.pi * j * k / N
            suma_k += np.exp(exponente) * v[j]
        v_k[k] = suma_k
    return v_k

v = np.array([1,2,3,4])

# Transformada propia
res_manual = transformada_fourier_discreta(v)

# Transformada de scipy (fft)
res_scipy = fft(v)# FFT

# Mostrar resultados
print("Resultado con tu función:")
print(res_manual)

print("\nResultado con np.fft.fft:")
print(res_scipy)

# Comparar numéricamente (diferencia absoluta por componente)
print("\nDiferencia absoluta:")
print(np.abs(res_manual - res_scipy))

#%%

# Ejercicio 15

def fft(x):
    '''Compute the DFT of an input x of N = 2**k samples'''
    
    # Get the length of the input
    N = len(x)
    
    # The DFT of a single sample is just the sample value itself
    # Nothing else to do here, so return
    if N == 1:
        return x
        
    else:
        # Recursively compute the even and odd DFTs
        X_even = fft(x[0::2])  # Start at 0 with steps of 2
        X_odd = fft(x[1::2])  # Start at 1 with steps of 2
        
        # Allocate the output array
        X = np.zeros(N, dtype=np.complex)
        
        # Combine the even and odd parts
        for m in range(N):
            # Find the alias of frequency m in the smaller DFTs
            m_alias = m % (N//2)
            X[m] = X_even[m_alias] + np.exp(-2j * np.pi * m / N) * X_odd[m_alias]
            
        return X


def fft_recursiva(x):
    """
    Implementación recursiva de la FFT usando el algoritmo de Cooley-Tukey.
    Asume que la longitud de x es una potencia de 2.
    """
    N = len(x)
    if N <= 1:
        return x
    else:
        # Dividir en partes par e impar
        x_par = fft_recursiva(x[::2])
        x_impar = fft_recursiva(x[1::2])

        # Calcular los factores twiddle
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        # Combinar resultados
        return np.concatenate([
            x_par + factor[:N//2] * x_impar,
            x_par - factor[:N//2] * x_impar
        ])

#%%

# Ejercicio 16

x = np.array([1.0, 2.0, 3.0, 4.0])  # Entrada real

# FFT completa
res_fft = fft(x)

# RFFT: solo la mitad
res_rfft = rfft(x)

print("FFT:", res_fft)
print("RFFT:", res_rfft)

#%%

# Ejercicio 19
"""
|            | 1209 Hz | 1336 Hz | 1477 Hz |
| ---------- | ------- | ------- | ------- |
| **697 Hz** | 1       | 2       | 3       |
| **770 Hz** | 4       | 5       | 6       |
| **852 Hz** | 7       | 8       | 9       |
| **941 Hz** | \*      | 0       | #       |

Ya tenemos las frecuencias de algunos números:

1 → 697 + 1209 Hz

3 → 697 + 1477 Hz

4 → 770 + 1209 Hz

5 → 770 + 1336 Hz

9 → 852 + 1477 Hz

0 → 941 + 1336 Hz

2 →  697 + 1336 Hz

6 → 770 + 1477 Hz

7 → 852  + 1209 Hz

8 → 852 + 1336 Hz
"""

#%%

# Ejercicio 21

def convolve(x, h):           # O(n^2)
    m, n = len(x), len(h)
    y = [0] * (m + n - 1)
    for i in range(m):
        for j in range(n):
            y[i + j] += x[i] * h[j]
    return y

def convolucion(v, w):
    return np.convolve(v, w, mode='full')   # convolve de numpy

#%%

# Ejercicio 22

def convolucion_fft_scipy(v, w):      # O(nlog(n))
    N = len(v) + len(w) - 1
    v_padded = np.pad(v, (0, N - len(v)))
    w_padded = np.pad(w, (0, N - len(w)))
    
    V = fft(v_padded)
    W = fft(w_padded)
    
    Y = V * W
    y = ifft(Y).real  # Nos quedamos con la parte real
    return y

#%%

# Ejercicio 23

# Filtro pasa bajos
v = np.array([0.25, 0.5, 0.25])

# Generar señal de ruido blanco
np.random.seed(0)
w = np.random.randn(1024)

# Convolucionar
y = np.convolve(w, v, mode='same')

# Obtener espectros
W = np.abs(fft(w))
Y = np.abs(fft(y))
freqs = fftfreq(len(w), d=1)  # Suponiendo frecuencia de muestreo normalizada

# Graficar
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(freqs[:512], W[:512])
plt.title("Espectro del ruido original")
plt.xlabel("Frecuencia")
plt.ylabel("Magnitud")

plt.subplot(1, 2, 2)
plt.plot(freqs[:512], Y[:512])
plt.title("Espectro después de filtrar con v")
plt.xlabel("Frecuencia")
plt.ylabel("Magnitud")

plt.tight_layout()
plt.show()

"""
ste es un filtro pasa-bajos (low-pass filter):

- Deja pasar las bajas frecuencias (variaciones lentas).

- Suprime las altas frecuencias (cambios bruscos o ruido).

Esto se debe a que suaviza la señal, promediando cada punto con sus vecinos.
"""
#%%

# Ejercicio 24

def derivar_f(f, x_vals):
    N = len(x_vals)
    
    # Transformada de Fourier
    tf = fft(f)
    
    # Vector de frecuencias (en orden no centrado)
    k = fftfreq(N, d=(x_vals[1] - x_vals[0])) * 2 * np.pi  # Frecuencias angulares
    
    # Reordenamos con fftshift
    tf_shifted = fftshift(tf)
    k_shifted = fftshift(k)
    
    tf_deriv_shifted = 1j * k_shifted * tf_shifted
    
    # Deshacemos el shift antes de aplicar la transformada inversa
    tf_deriv = ifftshift(tf_deriv_shifted)
    
    #  Transformada inversa: volvemos al dominio tiempo
    f_deriv = np.real(ifft(tf_deriv))

    return f_deriv # Devuelve la derivada en el dominio de Fourier


def f1(x):
    return np.sin(x) # x en [0,2π]

N1 = 64
L1 = 2 * np.pi  
x1_vals = np.linspace(0, L1, N1, endpoint=False)  
f1_vals = f1(x1_vals)
f1_deriv = derivar_f(f1_vals, x1_vals)

def f2(x):
    return np.cos(x)*(np.exp(-x**2))  # x en [-15,15]

N2 = 1024
L2 = 30  
x2_vals = np.linspace(-L2/2, L2/2, N2, endpoint=False)
f2_vals = f2(x2_vals)
f2_deriv = derivar_f(f2_vals, x2_vals)

def f3(x):
    x = np.array(x)
    return np.where((x >= 0) & (x <= 1), x, 0)

N3 = 1024
L3 = 1  
x3_vals = np.linspace(0, L3, N3, endpoint=False) 
f3_vals = f3(x3_vals)
f3_deriv = derivar_f(f3_vals, x3_vals)

plt.plot(x3_vals,f3_vals,label=("f3(x)"))
plt.plot(x3_vals,f3_deriv,label=("f3'(x)"))
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()


plt.plot(x1_vals,f1_vals,label=("f1(x)"))
plt.plot(x1_vals,f1_deriv,label=("f1'(x)"))
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()

plt.plot(x2_vals,f2_vals,label=("f2(x)"))
plt.plot(x2_vals,f2_deriv,label=("f2'(x)"))
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()

#%%

# Ejercicio 25

def integrar_f(f,x_vals):
    N = len(x_vals)
    dx = x_vals[1] - x_vals[0]

    # Transformada de Fourier
    tf = fft(f)

    # Vector de frecuencias angulares
    k = fftfreq(N, d=dx) * 2 * np.pi

    # Shift para centrar
    tf_shifted = fftshift(tf)
    k_shifted = fftshift(k)

    # Para evitar división por cero en k=0, podemos definir 1/(i*k) = 0 ahí
    with np.errstate(divide='ignore', invalid='ignore'):
        integrador = np.zeros_like(k_shifted, dtype=complex)
        integrador[k_shifted != 0] = 1 / (1j * k_shifted[k_shifted != 0])
        # En k=0 integrador=0, corresponde a constante de integración

    # Multiplicamos en el dominio de Fourier
    tf_int_shifted = tf_shifted * integrador

    # Deshacemos el shift
    tf_int = ifftshift(tf_int_shifted)

    # Transformada inversa: primitiva en el dominio tiempo
    f_int = np.real(ifft(tf_int))
    
    # Ajustar constante de integración para que coincida en x_vals[0]
    C = primitiva_f(x_vals[0]) - f_int[0]
    f_int += C

    return f_int

def f(x):
    return -3*(x**2)*np.exp(-x**3)

def primitiva_f(x):
    return np.exp(-x**3)

x_vals = np.linspace(-np.pi,np.pi,64)
mi_primitiva = integrar_f(f(x_vals),x_vals)

plt.plot(x_vals,mi_primitiva,label=("mi f'(x)"))
plt.plot(x_vals,primitiva_f(x_vals),label=("f'(x)"))
plt.legend()
plt.grid()
plt.show()





#%%

# GRAFICO DE FFT

# Parámetros
fs = 1000         # Frecuencia de muestreo (Hz)
T = 1.0           # Duración total (segundos)
N = int(fs * T)   # Número de muestras

# Tiempo
t = np.linspace(0, T, N, endpoint=False)

# Señal compuesta (esto es un conjunto de putnos f(x))
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t)

# Calcula la fft
fft_vals = fft(signal)
fft_mag = np.abs(fft_vals) / N  # Magnitud normalizada
freqs = fftfreq(N, d=1/fs)      # Vector de frecuencias

# Grafico
# Solo la mitad positiva del espectro
pos_mask = freqs >= 0
plt.plot(freqs[pos_mask], fft_mag[pos_mask])
plt.title("Espectro de la señal (FFT)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()



#%%

# UN EJEMPLO PARA ENTENDER


# 1. Definimos una función f(t) = seno + seno (combinación de frecuencias)
T = 1.0        # duración total del intervalo (en segundos)
N = 1000       # número de puntos
t = np.linspace(0.0, T, N, endpoint=False)
f = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

# 2. Calculamos la transformada de Fourier
F = np.fft.fft(f)
freq = np.fft.fftfreq(N, d=T/N)

# 3. Tomamos solo la mitad positiva de la frecuencia (por simetría)
half = N // 2
freq_pos = freq[:half]
F_pos = F[:half]

# 4. Reconstrucción aproximada con solo las componentes más fuertes
F_approx = np.zeros_like(F, dtype=complex)
dominant_freqs = np.argsort(np.abs(F))[-6:]  # top 6 componentes
F_approx[dominant_freqs] = F[dominant_freqs]
f_reconstructed = np.fft.ifft(F_approx).real

# 5. Graficamos todo
plt.figure(figsize=(15, 8))

# Original
plt.subplot(3, 1, 1)
plt.plot(t, f)
plt.title("1. Señal original f(t)")
plt.xlabel("Tiempo [s]")

# Transformada de Fourier
plt.subplot(3, 1, 2)
plt.stem(freq_pos, np.abs(F_pos), use_line_collection=True)
plt.title("2. Magnitud de la Transformada de Fourier |F(ω)|")
plt.xlabel("Frecuencia [Hz]")

# Reconstrucción
plt.subplot(3, 1, 3)
plt.plot(t, f_reconstructed, label='Aproximación')
plt.plot(t, f, '--', alpha=0.5, label='Original')
plt.title("3. Reconstrucción con frecuencias dominantes")
plt.xlabel("Tiempo [s]")
plt.legend()

plt.tight_layout()
plt.show()

"""
Un pico en 5 Hz fft → f(t) contiene una onda seno que oscila 5 veces por segundo.

Un pico en 20 Hz en la fft → f(t) contiene una onda seno que oscila 20 veces por segundo.
"""
#%%

# Parámetros
fs = 1000          # frecuencia de muestreo en Hz
T = 1/fs           # periodo de muestreo
N = 1024           # número de muestras
t = np.arange(N) * T  # vector de tiempo

# Señal de ejemplo: una suma de dos senos
f1 = 50    # frecuencia 1 en Hz
f2 = 120   # frecuencia 2 en Hz
x = 0.7*np.sin(2*np.pi*f1*t) + 0.3*np.sin(2*np.pi*f2*t)

# Transformada de Fourier
X = np.fft.fft(x)

# Frecuencias asociadas
freqs = np.fft.fftfreq(N, d=T)

# Magnitud de la transformada (normalizada)
X_mag = np.abs(X) / N

# Graficar la magnitud (solo la mitad positiva)
plt.plot(freqs[:N//2], X_mag[:N//2])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Espectro de frecuencia')
plt.grid()
plt.show()


# GUIA 1 DE MODELADO CONTINUO 

import numpy as np
import matplotlib.pyplot as plt

#%% 

# Ejercicio 1

# Método de Euler
def euler_method(lambda_, h, t_max):
    t_values = np.arange(0, t_max + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = 1  

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i - 1] + h * lambda_ * y_values[i - 1]
    
    return t_values, y_values

# Solución exacta
def exact_solution(lambda_, t_values, y0=1):
    return y0 * np.exp(lambda_ * t_values)

# Parámetros
lambda_values = [1, -1, 5, -5]  
h_values = [0.1, 0.5, 1.5] 
t_max = 5

# Graficar soluciones para distintos lambda y h
plt.figure(figsize=(12, 8))

for lambda_ in lambda_values:
    plt.subplot(2, 2, lambda_values.index(lambda_) + 1)
    t_exact = np.linspace(0, t_max, 1000)
    y_exact = exact_solution(lambda_, t_exact)
    plt.plot(t_exact, y_exact, 'k-', label="Solución exacta")

    for h in h_values:
        t_euler, y_euler = euler_method(lambda_, h, t_max)
        plt.plot(t_euler, y_euler, marker='o', linestyle='--', label=f'Euler h={h}')

    plt.title(f"$\\lambda = {lambda_}$")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

#%%

# Ejercicio 2

# Método de Euler para la ecuación logística
def logistic_euler(r, k, P0, t_max, h):
    t_values = np.arange(0, t_max + h, h)
    P_values = np.zeros_like(t_values)
    P_values[0] = P0
    
    for i in range(1,len(t_values)):
        P_values[i] = P_values[i-1] + h * r * P_values[i-1] * (1 - P_values[i-1] / k)
        
    return t_values, P_values

# Solución exacta
def logistic_solution(r, k, P0, t_values):
    return k / (1 + ((k-P0)/P0)*np.exp(-r*t_values))

# Parámetros
r = 0.5 
K = 100  
t_max = 20
P0_values = [10, 50, 120] 
h_values = [0.1, 0.5, 1.0]  

#Graficar soluciones para distintos P0 y h
plt.figure(figsize=(12, 8))

t_exact = np.linspace(0, t_max, 1000)
for P0 in P0_values:
    plt.subplot(2, 2, P0_values.index(P0) + 1)
    y_exact = logistic_solution(r, K, P0, t_exact)
    plt.plot(t_exact, y_exact, 'k-', label="Solución exacta")
    
    for h in h_values:
        t_euler, y_euler = logistic_euler(r, K, P0, h, t_max)
        plt.plot(t_euler, y_euler, marker='o', linestyle='--', label=f'Euler h={h}')
    
    plt.title(f"P0 = {P0}")
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

#%%

# Ejercicio 3






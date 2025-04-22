# GUIA 1 DE MODELADO CONTINUO 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

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

# Punto C (lo hice independiente a los incisos anteriores)

from scipy.stats import linregress

def euler_method(f, h, t_max, y0):
    t_values = np.arange(0, t_max + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1])
    
    return t_values, y_values

def extended_euler(f, h, t_max, y0):
    t_values = np.arange(0, t_max + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t_prev, y_prev = t_values[i - 1], y_values[i - 1]
        k1 = f(t_prev, y_prev)
        k2 = f(t_prev + h / 2, y_prev + (h / 2) * k1)
        y_values[i] = y_prev + h * k2
    
    return t_values, y_values

def exact_solution(t, lambda_):
    return np.exp(lambda_ * t)

def rutina(f, lambda_, t_max, M, y0=1):
    error1, error2, h_values = [], [], []
    
    for m in range(1, M):  # Evitamos m=0 para evitar h demasiado grande
        h = 2**(-m)
        h_values.append(h)
        
        t1, y1 = euler_method(f, h, t_max, y0)
        t2, y2 = extended_euler(f, h, t_max, y0)
        
        y_exact = exact_solution(t1[-1], lambda_)
        
        e1 = abs(y1[-1] - y_exact)
        e2 = abs(y2[-1] - y_exact)

        error1.append(e1)
        error2.append(e2)
    
    log_h = np.log(h_values)
    log_e1 = np.log(error1)
    log_e2 = np.log(error2)
    
    # Eliminamos valores extremos para evitar problemas numéricos
    valid_range = slice(2, -2)  # Ajusta según la estabilidad
    slope1, _, _, _, _ = linregress(log_h[valid_range], log_e1[valid_range])
    slope2, _, _, _, _ = linregress(log_h[valid_range], log_e2[valid_range])
    
    plt.figure(figsize=(8, 5))
    plt.plot(log_h, log_e1, 'o-', label=f"Euler (orden ~ {slope1:.2f})")
    plt.plot(log_h, log_e2, 's-', label=f"Euler modificado (orden ~ {slope2:.2f})")
    plt.xlabel("log(h)")
    plt.ylabel("log(error)")
    plt.title("Orden de convergencia en log-log")
    plt.legend()
    plt.grid()
    plt.show()

# Parámetros
def f(t, y):
    return -2 * y  # Lambda = -2

lambda_ = -2
t_max = 5
M = 10

rutina(f, lambda_, t_max, M)

#%%

# Ejercicio 3

# Parámetros
K = 1000
m = 0.1
t_max = 50  # años

# Funciones
def r(t):
    return 0.2 + 0.2 * np.cos(2 * np.pi * t)

def f(t, y):
    return r(t) * y * (1 - y / K) - m * y

# Método de Heun propuesto
def animales_silvestres(h,y0):
    t_values = np.arange(0,t_max + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(1,len(t_values)):
        
        y_values[i] = y_values[i-1] + (h/4)*(f(t_values[i-1],y_values[i-1]) + 3*f(t_values[i-1]+ (2/3)*h,y_values[i-1] + (2/3)*h*f(t_values[i-1],y_values[i-1])))

    return y_values, t_values


estimacion1 , tiempos= animales_silvestres(1/365, 100)
estimacion2, _ = animales_silvestres(1/365, 500)
estimacion3 , _ = animales_silvestres(1/365, 1000)

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(tiempos, estimacion1, label="y₀ = 100")
plt.plot(tiempos, estimacion2, label="y₀ = 500")
plt.plot(tiempos, estimacion3, label="y₀ = 1000")
plt.xlabel("Tiempo (años)")
plt.ylabel("Población")
plt.title("Evolución de la población de animales silvestres")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%

# Ejercicio 4

def rungekutta(y0,h,t_max):
    t_values = np.arange(0, t_max + h, h)
    n = len(t_values)
    k = len(y0)
    y_values = np.zeros((k,n))
    y_values[:,0] = y0

    for i in range(1,len(t_values)):
        k1 = f(t_values[i-1], y_values[:,i-1])
        k2 = f(t_values[i-1] + (1/2)*h,y_values[:,i-1] + (1/2)*h*k1)
        k3 = f(t_values[i-1] + (1/2)*h,y_values[:,i-1] + (1/2)*h*k2)
        k4 = f(t_values[i-1] + h,y_values[:,i-1] + h*k3)
            
        y_values[:,i]= y_values[:,i-1] + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)
        
    return t_values ,y_values

def f(t, y):  # Lotka-Volterra
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
    x, y_ = y
    dxdt = alpha * x - beta * x * y_
    dydt = delta * x * y_ - gamma * y_
    return np.array([dxdt, dydt])

# Condiciones iniciales
y0 = np.array([10, 5])  # x=10, y=5
h = 0.01
t_max = 10

t_vals, Y = rungekutta(y0, h, t_max)

# Graficar resultados
import matplotlib.pyplot as plt

plt.plot(t_vals, Y[0], label='x (presa)')
plt.plot(t_vals, Y[1], label='y (depredador)')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelo de Lotka-Volterra (Runge-Kutta 4)')
plt.legend()
plt.grid(True)
plt.show()


#%%

#Ejercicio 6

def Lotka_Volterra(y0,x0,t_max,alpha,beta,gamma,delta,h=0.01):
    t_values = np.arange(0,t_max+h,h)
    y_values = np.zeros_like(t_values)
    x_values = np.zeros_like(t_values)
    y_values[0] = y0
    x_values[0] = x0
    
    for i in range(1,len(t_values)):
        
        x_values[i] = x_values[i-1] + h*(-alpha*x_values[i-1] + gamma*x_values[i-1]*y_values[i-1])
        y_values[i] = y_values[i-1] + h*(beta*y_values[i-1] -delta*y_values[i-1]*x_values[i-1]) 

    return t_values, y_values, x_values

alpha = 0.25
beta =1
gamma =0.01
delta =0.01
t_max =50
x0= 80
y0= 30

t_values, y_values, x_values = Lotka_Volterra(y0, x0, t_max, alpha, beta, gamma, delta)

plt.plot(t_values, x_values, label = "Depredadores", color = "red")
plt.plot(t_values, y_values, label = "Presas", color="green")
plt.xlabel("Tiempo")
plt.ylabel("Poblacion")
plt.title("Modelo Depredador-Presa con Euler")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(x_values, y_values, color='purple')
plt.xlabel('Depredadores (x)')
plt.ylabel('Presas (y)')
plt.title('Diagrama de fases: Presas vs Depredadores')
plt.grid()
plt.show()

#%%

# Ejercicio 9

def Hutchinson_euler(r, k, h, t_max, tau, N0 = 0.5):
    t_values = np.arange(0, t_max + h, h)
    N_values = np.zeros_like(t_values)
    N_values[0] = N0

    for i in range(1,len(t_values)):
        if (i- tau) < 0 :
            N_values[i] = N0
        else :
            N_values[i] = N_values[i-1] + h * r * N_values[i-1] * (1-(N_values[i-1-tau]/k))
        
    return t_values , N_values

# Parámetros
r = 0.5
tau_values = [1, 5, 10]  
k = 100
t_max = 50
h = 0.1

# Funcíon de Hutchinson sin delay
def Hutchinson_euler_sin_delay(r, k, h, t_max, N0 = 0.5):
    t_values = np.arange(0, t_max + h, h)
    N_values = np.zeros_like(t_values)
    N_values[0] = N0

    for i in range(1,len(t_values)):
        N_values[i] = N_values[i-1] + h * r * N_values[i-1] * (1-(N_values[i-1]/k))
        
    return t_values , N_values


#Graficar soluciones para distintos tau y h
plt.figure(figsize=(12, 8))
t_sin_delay , y_sin_delay = Hutchinson_euler_sin_delay(r, k, h, t_max)
plt.plot(t_sin_delay, y_sin_delay, marker='o', linestyle='--')
for tau in tau_values:
    t_euler , y_euler = Hutchinson_euler(r, k, h, t_max, tau)
    plt.plot(t_euler, y_euler, marker='o', linestyle='--', label=f'tau={tau}')

    plt.title("Solucion de la ecuacion de Hutchinson con método de Euler")
    plt.xlabel("t")
    plt.ylabel("N(t)")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()


#%%

# Ejercicio 11

# Constantes físicas
G = 0.4982  
mS = 1.9891e30
mT = 5.97e24
mL = 7.3477e22
masses = [mS, mT, mL]

def tres_cuerpos(t_max,h,u0):
    t_values = np.arange(0,t_max + h, h )    
    u_values = np.zeros((len(t_values),3,4))
    u_values[0] = u0
    
    for t in range(1,len(t_values)):
        for i in range(3):
            xi, yi = u_values[t-1, i, 2] , u_values[t-1, i, 3]
            vxi ,vyi = u_values[t-1, i, 0] , u_values[t-1, i, 1]
            
            # Inicializar aceleración
            axi, ayi = 0.0, 0.0

            for j in range(3):
                if j == i : 
                    continue
                xj , yj = u_values[t-1, j, 0], u_values[t-1, j, 1]
                dx , dy = xj - xi, yj - yi
                dist_sq = dx**2 + dy**2
                dist_cubed = dist_sq * np.sqrt(dist_sq)
                axi += G * masses[j] * dx/dist_cubed
                ayi += G * masses[j] * dy/dist_cubed
                
            # Método de Euler
            new_x = xi + h * vxi
            new_y = yi + h * vyi
            new_vx = vxi + h * axi
            new_vy = vyi + h * ayi
            
            u_values[t, i] = [new_x, new_y, new_vx, new_vy]
            
    return t_values , u_values

# Posiciones iniciales
dTS = 1.49597887e11  # m
dTL = 3.844e8        # m

vT = 2 * np.pi * dTS / 365
vLT = 2 * np.pi * dTL / 28
vLS = vT

# Estado inicial: [x, y, vx, vy] para S, T, L
u0 = np.array([
    [0,     0,     0,    0],       # Sol
    [dTS,   0,     0,    vT],      # Tierra
    [dTS, dTL,  -vLT,    vLS]      # Luna
])

# Simulación
t_max = 365      # días
h = 1            # paso de 1 día

t_vals, u_vals = tres_cuerpos(t_max, h, u0)

xS, yS = u_vals[:, 0, 0], u_vals[:, 0, 1]
xT, yT = u_vals[:, 1, 0], u_vals[:, 1, 1]
xL, yL = u_vals[:, 2, 0], u_vals[:, 2, 1]

plt.figure(figsize=(8, 8))
plt.plot(xS, yS, label='Sol', color='orange')
plt.plot(xT, yT, label='Tierra', color='blue')
plt.plot(xL, yL, label='Luna', color='gray')
plt.scatter(xS[0], yS[0], color='orange', s=50)
plt.legend()
plt.axis('equal')
plt.title("Sistema Sol-Tierra-Luna")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.show()

#%%

# Ejercicio 14

# Paso 1: Definimos el sistema con parámetros verdaderos para simular datos
def sistema(t,z,alpha,gamma,beta,delta,epsilon):
    x, y = z
    dx = -alpha*x + gamma*x*y
    dy = beta*y - delta*x*y - epsilon*y**2
    return [dx,dy]

# Parámetros
true_params = [1,0.2,1.0,0.01,0.02]
z0 = [40,9]
t_span = (0,50)
t_eval = np.linspace(*t_span,100)

sol = solve_ivp(sistema,t_span,z0,args=tuple(true_params),t_eval = t_eval)
x_data , y_data = sol.y
t_data = sol.t

# Estimamos derivadas numéricas
dx_data = np.gradient(x_data, t_data)
dy_data = np.gradient(y_data, t_data)

# Paso 2: Ajuste de parámetros
def loss(params):
    alpha, gamma, beta, delta, epsilon = params
    dx_model = -alpha*x_data + gamma*x_data * y_data
    dy_model = beta * y_data - delta * x_data * y_data - epsilon * y_data**2
    return np.mean((dx_model - dx_data)**2 + (dy_model - dy_data)**2)

# Paso 3: Minimización del error cuadrático
initial_guess = [1, 0.1, 1.2, 2, 0.01]
result = minimize(loss, initial_guess, bounds=[(0, 2)]*5)
fitted_params = result.x

# Resultados
param_names = ['alpha', 'gamma', 'beta', 'delta', 'epsilon']
print("\nParámetros ajustados:")
for name, val, true in zip(param_names, fitted_params, true_params):
    print(f"{name}: ajustado = {val:.4f} (real = {true})")

# Paso 4: Comparación visual
sol_fit = solve_ivp(sistema, t_span, z0, args=tuple(fitted_params), t_eval=t_eval)

plt.figure(figsize=(10,5))
plt.plot(t_eval, x_data, 'r-', label='Presas (simulado)')
plt.plot(t_eval, y_data, 'b-', label='Depredadores (simulado)')
plt.plot(t_eval, sol_fit.y[0], 'r--', label='Presas (ajustado)')
plt.plot(t_eval, sol_fit.y[1], 'b--', label='Depredadores (ajustado)')
plt.legend()
plt.title("Ajuste del modelo predador-presa con competencia intraespecífica")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.grid()
plt.tight_layout()
plt.show()

#%%

# Ejercicio 16

def bouncing_ball(v0,y0,t_max,h):
    t_values = np.arange(0,t_max + h, h )
    y_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)
    
    y_values[0] = y0
    v_values[0] = v0
    
    for i in range(1,len(t_values)):
        v_values[i] = v_values[i-1] - 10*h
        y_values[i] = y_values[i-1] + h* v_values[i-1]
        
        if y_values[i] < 0:
            y_values[i] = 0
            v_values[i] = -0.8*v_values[i]
            
    return t_values, y_values, v_values 
    
# Parámetros
v0=0
y0=10
t_max = 50
h = 0.001

# Graficar

t_euler, y_euler, v_euler = bouncing_ball(v0, y0, t_max, h)

plt.figure(figsize=(12, 8))

# Graficar la posición (y)
plt.plot(t_euler, y_euler, label='Posición (y)', color='b', marker='o', linestyle='--')


plt.title("Solución de Bouncing Ball con Método de Euler")
plt.xlabel("Tiempo (t)")
plt.ylabel("Posición (y) ")
plt.legend()
plt.grid()

#%%

# Ejercicio 21

def sistema(t,z,u):
    x , v  =  z
    dxdt = v
    dvdt = u * (1-x**2)*v - x
    return [dxdt, dvdt]


# Parámetros
u_values = [-1,0,1]
t_span = (0,20)
t_eval = np.linspace(*t_span,1000)
z0_values = [[1, 0], [0.5, 0.5], [2, 0]]


plt.figure(figsize=(12,8))
plt.subplot(1, 2, 1)

for z0 in z0_values:
    for u in u_values:
        sol = solve_ivp(sistema, t_span, z0, args=(u,), t_eval = t_eval,rtol=1e-6, atol=1e-9)
        x = sol.y[0]
        v = sol.y[1]
        
        label = f"$u={u}$, $z_0={z0}$"
        plt.plot(x,v,label=label)
        
plt.xlabel("$x$")
plt.ylabel("$x'$")
plt.title("Trayectoria en el plano de fases")
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
        
#%%

# Ejercicio 22

def pendulo(t,z,A,b):
    theta, v = z
    dthetadt = v
    dvdt = -A*np.sin(theta) - b*v
    return [dthetadt,dvdt]


def pendulo_armonico(t,z,A,b):
    theta, v = z
    dthetadt = v
    dvdt = -A*theta - b*v
    return [dthetadt,dvdt]

# Parámetros

z0 = [1,1]
A = 5
b = -1
t_span =(0,50)
t_eval = np.linspace(*t_span,1000)

sol = solve_ivp(pendulo,t_span,z0,t_eval = t_eval, args=(A,b))
sol2 = solve_ivp(pendulo_armonico,t_span,z0,t_eval = t_eval, args=(A,b))

plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
plt.plot(sol.y[0],sol.y[1],label="Trajectoria no Armonico")
plt.xlabel("theta")
plt.ylabel("v")

plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(sol2.y[0],sol2.y[1],label="Trajectoria Armonico")
plt.xlabel("theta")
plt.ylabel("v")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%%

# Ejercicio 23

def Lorentz(t,z0,r):
    sigma = 10
    b = 8/3
    
    x,y,z = z0
    dxdt = sigma*(y-x)
    dydt = r*x - y - x*z
    dzdt = x*y - b*z
    
    return [dxdt,dydt,dzdt]

# Parámetros

r_values = [1/2,1,2,15,25]
z0 = [1,1,1]
t_span = (0,50)
t_eval = np.linspace(*t_span,1000)


fig = plt.figure(figsize=(20,10))

for i, r in enumerate(r_values):
    sol = solve_ivp(Lorentz, t_span, z0, t_eval=t_eval, args=(r,))
    
    ax = fig.add_subplot(2, 3, i+1, projection='3d')  # 3D subplot
    ax.plot(sol.y[0], sol.y[1], sol.y[2], label=f"r = {r}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Diagrama de fases para r = {r}")
    ax.legend()

plt.tight_layout()
plt.show()
    
    
    


















import numpy as np
import matplotlib.pyplot as plt
import scipy 

#%%

# Ejercicio 1
# quiero matrices D tales que u' = D * u

def forward_diff_matrix(n,h, periodic = True):
    D = np.zeros((n,n))
    for i in range(n-1):
        D[i,i] = -1/h
        D[i,i+1] = 1/h
    if periodic:
        D[n - 1, n - 1] = -1 / h
        D[n - 1, 0] = 1 / h
    return D

def backward_diff_matrix(n,h, periodic = True):
    D = np.zeros((n,n))
    for i in range(1,n):
        D[i,i] = 1/h
        D[i,i-1] = -1/h
    if periodic:
        D[0, 0] = 1 / h
        D[0, n-1] = -1 / h
    return D

def centered_diff_matrix(n,h, periodic = True):
    D = np.zeros((n,n))
    for i in range(1,n-1):
        D[i,i+1] = 1/(2 * h)
        D[i,i-1] = -1/(2 * h)
    if periodic:
        D[0, n - 1] = -1 / (2 * h)
        D[0, 1] = 1 / (2 * h)
        D[n - 1, n - 2] = -1 / (2 * h)
        D[n - 1, 0] = 1 / (2 * h)
    return D

# Ejemplo de uso:
    
a, b = 0, 2*np.pi
n = 100
h = (b - a) / n

D = centered_diff_matrix(n, h)

x = np.linspace(a, b, n, endpoint=False)
u = np.sin(x)

u_prime_approx = D @ u
u_prime_exact = np.cos(x)

error = np.linalg.norm(u_prime_approx - u_prime_exact, ord=np.inf)
print(f"Error máximo: {error:.3e}")
#%%

# Ejercicio 2

# Funciones y sus derivadas exactas
def u(x):
    return np.log(x + 1) - np.log(2)*x

def u_prime(x):
    return 1 / (x + 1) - np.log(2)

def v(x):
    return np.exp(np.cos(2 * np.pi * x))

def v_prime(x):
    return -2 * np.pi * np.sin(2 * np.pi * x) * np.exp(np.cos(2 * np.pi * x))


# Funcion para computar los errores 
def compute_errors(f, f_prime, diff_matrix_func, periodic):
    hs = []
    errors = []
    
    for i in range(3, 9):  # i = 3,...,8
        n = 2 ** i
        h = 1.0 / n
        x = np.linspace(0, 1, n, endpoint=False if periodic else True)
        
        D = diff_matrix_func(n, h, periodic)
        u_vals = f(x)
        u_deriv_approx = D @ u_vals
        u_deriv_exact = f_prime(x)
        
        error = np.linalg.norm(u_deriv_approx - u_deriv_exact, np.inf)
        hs.append(h)
        errors.append(error)
    
    return np.array(hs), np.array(errors)


# Graficar en escala log-log

# Comparar centrada con y sin periodicidad para v(x)
hs1, errs1 = compute_errors(v, v_prime, centered_diff_matrix, periodic=False)
hs2, errs2 = compute_errors(v, v_prime, centered_diff_matrix, periodic=True)

plt.plot(np.log(hs1), np.log(errs1), 'o-', label="sin períodica")
plt.plot(np.log(hs2), np.log(errs2), 's--', label="con períodica")
plt.legend()
plt.xlabel("log(h)")
plt.ylabel("log(error)")
plt.title("Error centrada en v(x): con/sin condiciones periódicas")
plt.grid(True)
plt.show()

hs3, errs3 = compute_errors(u, u_prime, centered_diff_matrix, periodic=True)

plt.plot(np.log(hs3), np.log(errs3), 's--', label="con períodica")
plt.legend()
plt.xlabel("log(h)")
plt.ylabel("log(error)")
plt.title("Error centrada en u(x): con condiciones periódicas")
plt.grid(True)
plt.show()

#%%

# Ejercicio 3

def resolver_poisson_1D(f, alpha, beta, m):
    # Paso 1: Definición de malla
    h = 1.0 / (m + 1)
    x = np.linspace(0, 1, m + 2)  # incluye 0 y 1
    x_interior = x[1:-1]

    # Paso 2: Vector del lado derecho ajustado con condiciones de borde
    b = f(x_interior)
    b[0] -= alpha / h**2
    b[-1] -= beta / h**2

    # Paso 3: Matriz tridiagonal A
    A = np.zeros((m, m))
    for i in range(m):
        A[i, i] = -2 / h**2
        if i > 0:
            A[i, i - 1] = 1 / h**2
        if i < m - 1:
            A[i, i + 1] = 1 / h**2

    # Paso 4: Resolver el sistema
    u_interior = np.linalg.solve(A, b)

    # Paso 5: Agregar condiciones de borde
    u = np.zeros(m + 2)
    u[0] = alpha
    u[-1] = beta
    u[1:-1] = u_interior

    return x, u

# Ejemplo de uso
f = lambda x: np.sin(np.pi * x)
alpha = 0
beta = 0
m = 10

x, u = resolver_poisson_1D(f, alpha, beta, m)

# Graficar la solución
plt.plot(x, u, 'o-', label='Solución numérica')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solución de u\'\'=f(x) con condiciones de Dirichlet')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Ejercicio 4

def construir_matriz_A_neumann_dirichlet(m, h):
    A = np.zeros((m, m))
    
    # Primera fila con condición de Neumann (aproximación forward)
    A[0, 0] = -1 / h**2
    A[0, 1] = 1 / h**2

    # Filas internas
    for i in range(1, m-1):
        A[i, i-1] = 1 / h**2
        A[i, i] = -2 / h**2
        A[i, i+1] = 1 / h**2

    # Última fila con Dirichlet (u(1) = 0, no se incluye u_{m+1})
    A[m-1, m-2] = 1 / h**2
    A[m-1, m-1] = -2 / h**2

    return A

def resolver_poisson_neumann_dirichlet(f, m):
    h = 1 / (m + 1)
    x = np.linspace(h, 1 - h, m)  # nodos internos
    b = f(x)

    A = construir_matriz_A_neumann_dirichlet(m, h)
    
    # Resolver el sistema A u = b
    u_sol = np.linalg.solve(A, b)

    # Agregar valores en los extremos: Neumann en x=0, Dirichlet en x=1
    u = np.zeros(m + 2)
    u[1:-1] = u_sol
    u[-1] = 0  # Dirichlet

    x_full = np.linspace(0, 1, m + 2)
    return x_full, u

# ------------------------------
# Ejemplo de uso
# ------------------------------

f = lambda x: np.sin(np.pi * x)  # función f(x)
m = 20  # cantidad de puntos internos

x, u = resolver_poisson_neumann_dirichlet(f, m)

# Graficar la solución
plt.plot(x, u, label='Solución numérica')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solución de u\'\'=f(x) con Neumann en x=0 y Dirichlet en x=1')
plt.grid(True)
plt.legend()
plt.show()

#%%

# Ejercicio 5

# Parámetros
L = 1                     # Longitud desde -1 a 1
x0, x1 = -L, L
T = 0.1                   # Tiempo total de simulación
dx = 0.05
dt = 0.0012               # Cambiar a 0.0013 para ver inestabilidad
r = dt / dx**2

# Discretización
x = np.arange(x0, x1 + dx, dx)
N = len(x)
timesteps = int(T / dt)

# Condición inicial (a)
def u0_a(x):
    return np.where(x < 0, x + 1, 1 - x)

# Condición inicial (b)
def u0_b(x):
    return np.sin(np.pi * x)

# Elegir condición inicial:
u = u0_b(x)               # cambiar a u0_b(x) para caso (b)

# Aplicar condiciones de frontera (Dirichlet homogéneas)
u[0] = 0
u[-1] = 0

# Construir matriz A (tridiagonal)
A = np.diag((1 - 2*r) * np.ones(N - 2)) + \
    np.diag(r * np.ones(N - 3), k=1) + \
    np.diag(r * np.ones(N - 3), k=-1)

# Guardar evolución para graficar
historia = [u.copy()]

# Evolución en el tiempo
for _ in range(timesteps):
    u_interior = u[1:-1]                    # no incluye los bordes
    u_new = A @ u_interior
    u[1:-1] = u_new                         # actualizamos interior
    historia.append(u.copy())              # guardar para graficar

# Convertir historia a matriz para graficar
historia = np.array(historia)

# Graficar solución
X, T = np.meshgrid(x, np.linspace(0, T, len(historia)))
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, T, historia, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Evolución de la ecuación del calor')
plt.show()


#%%

# Ejercicio 6


# Parámetros
L = 1
x0, x1 = -L, L
T = 0.1
dx = 0.05
dt = 0.0012
r = dt / dx**2

# Discretización
x = np.arange(x0, x1 + dx, dx)
N = len(x)
timesteps = int(T / dt)

# Condición inicial
def u0_b(x):
    return np.sin(np.pi * x)

u = u0_b(x)
u[0] = 0
u[-1] = 0

# Función fuente f(x, t)
def fuente(x, t):
    return np.exp(-t) * np.sin(np.pi * x)  # ejemplo arbitrario

# Matriz A
A = np.diag((1 - 2*r) * np.ones(N - 2)) + \
    np.diag(r * np.ones(N - 3), k=1) + \
    np.diag(r * np.ones(N - 3), k=-1)

# Evolución
historia = [u.copy()]
for n in range(timesteps):
    t = n * dt
    u_interior = u[1:-1]
    f_interior = fuente(x[1:-1], t)
    u_new = A @ u_interior + dt * f_interior
    u[1:-1] = u_new
    historia.append(u.copy())

# Gráfica
historia = np.array(historia)
X, Tm = np.meshgrid(x, np.linspace(0, T, len(historia)))
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Tm, historia, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Evolución con fuente f(x,t)')
plt.show()


#%%

# Ejercicio 9 (METODO DE LINEAS)

# Ecuacion de calor 1D [-1,1]

from scipy.integrate import solve_ivp

# Parámetros
alpha = 1.0
a, b = 0.0, 0.0           # condiciones de frontera
T = 0.1                   # tiempo final
N = 20                    # número de puntos internos
h = 2 / (N + 1)           # paso espacial
x = np.linspace(-1 + h, 1 - h, N)  # puntos interiores

# Condición inicial: campana gaussiana
def g(x):
    return np.exp(-10 * x**2)

u0 = g(x)

# Definimos el sistema de EDOs
def heat_eq(t, u):
    dudt = np.zeros_like(u)
    for i in range(N):
        u_ip1 = b if i == N-1 else u[i+1]
        u_im1 = a if i == 0 else u[i-1]
        dudt[i] = alpha * (u_ip1 - 2*u[i] + u_im1) / h**2
    return dudt

# Resolviendo con solve_ivp
sol = solve_ivp(heat_eq, [0, T], u0, t_eval=np.linspace(0, T, 100))

# Graficar solución en distintos tiempos
from matplotlib import cm

X = x
T_grid = sol.t
U = sol.y  # Cada columna es el estado en un tiempo

fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')
T_mesh, X_mesh = np.meshgrid(T_grid, X)

ax.plot_surface(X_mesh, T_mesh, U, cmap=cm.viridis)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.title("Solución de la ecuación del calor por el método de líneas")
plt.show()


#%%


# Ejercicio 11 (METODO DE LINEAS )

# Ecuacion de calor 2D  [-1,1]x[-1,1]

# Parámetros
alpha = 1.0
T = 0.1
N = 20
h = 2 / (N + 1)
x = np.linspace(-1 + h, 1 - h, N)
y = np.linspace(-1 + h, 1 - h, N)
X, Y = np.meshgrid(x, y, indexing='ij')  # grilla para condición inicial

# Condición inicial: campana gaussiana 2D
def g(x, y):
    return np.exp(-10 * (x**2 + y**2))

U0 = g(X, Y).reshape(-1)  # vectorizamos la condición inicial

# Sistema de EDOs para el método de líneas en 2D
def heat_eq_2d(t, u):
    u = u.reshape((N, N))  # reshape a matriz 2D
    dudt = np.zeros_like(u)
    for i in range(N):
        for j in range(N):
            u_ip1 = 0 if i == N-1 else u[i+1, j]
            u_im1 = 0 if i == 0   else u[i-1, j]
            u_jp1 = 0 if j == N-1 else u[i, j+1]
            u_jm1 = 0 if j == 0   else u[i, j-1]
            dudt[i, j] = alpha * (u_ip1 - 2*u[i,j] + u_im1 + u_jp1 - 2*u[i,j] + u_jm1) / h**2
    return dudt.reshape(-1)

# Resolver el sistema de ODEs
tiempos = np.linspace(0, T, 100)
sol = solve_ivp(heat_eq_2d, [0, T], U0, t_eval=tiempos)

# Graficar la solución en el último tiempo

U_final = sol.y[:, -1].reshape(N, N)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
X_plot, Y_plot = np.meshgrid(x, y, indexing='ij')
ax.plot_surface(X_plot, Y_plot, U_final, cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y,T)')
plt.title("Solución de la ecuación del calor en 2D (t = T)")
plt.show()




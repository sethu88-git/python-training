import matplotlib.pyplot as plt
import numpy as np


def derivative(coeff):
    der = []
    n = len(coeff)-1
    for i in range(n):
        der.append(coeff[i]*(n-i))
    return der


def eval_fn(coeff, x):
    n = len(coeff)
    sum = 0.0
    for i in range(n):
        sum += coeff[i] * x**(n-1-i)
    return sum


def grad_desc(coeff, der, rate):
    real_roots = check_real_roots(der)
    if len(real_roots) == 0:
        raise ValueError
    new_minimum = init_min(coeff, real_roots)
    path = [new_minimum]
    iterations = 0
    while (True):
        iterations += 1
        minimum = new_minimum
        grad = eval_fn(der, minimum)
        alpha = min(rate, 1 / (1 + abs(grad)))
        if abs(grad) > 10**6:
            raise ValueError
        new_minimum = minimum - grad*alpha
        path.append(new_minimum)
# %%
        if abs((new_minimum-minimum)/minimum) < 10**-9:

            return (new_minimum, path)
            break
        if (iterations > 100000 or abs((new_minimum-minimum)/minimum) > 10**6):
            raise ValueError


def check_real_roots(der):
    roots = np.roots(der)
    real_roots = [root for root in roots if np.isreal(root)]
    return (real_roots)


def init_min(coeff, x):
    inp = np.array(x)
    out = np.array([eval_fn(coeff, i) for i in inp])
    return (inp[np.argmin(out)])


def plot_graph(x, path, coeff):
    x_vals = np.linspace(x - 2, x + 2, 500)
    y_vals = [eval_fn(coeff, xi) for xi in x_vals]
    plt.plot(x_vals, y_vals, label='f(x)', linewidth=2)
    traj_y = [eval_fn(coeff, px) for px in path]
    plt.scatter(path, traj_y, color='orange', s=10, label='Descent Path', zorder=4)
    plt.scatter(x, eval_fn(coeff, x), color='red', label='Estimated Minimum', zorder=5)
    plt.annotate(f"Min at x={x:.4f}", (x, eval_fn(coeff, x)),
                 textcoords="offset points", xytext=(-10, 10), ha='center', fontsize=9, color='red')
    plt.title("Gradient Descent Minimum")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


n = int(input("Enter the order of function: "))
coeff = []
print("Enter coeffiecients of x^n, x^(n-1)... x, k seperated by return:")
for _ in range(n+1):
    coeff.append(float(input().strip()))
try:
    x, path = grad_desc(coeff, derivative(coeff), 0.001)
except ValueError:
    print("Convergence not reached")
else:
    print(f"Minimum {eval_fn(coeff, x)} reached at {x}")
    plot_graph(x, path, coeff)

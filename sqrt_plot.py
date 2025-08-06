import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(-2.0, 2.0, 1000)
    y1 = np.sqrt(4-x**2)
    y2 = -np.sqrt(4-x**2)
    plt.plot(x, y1, color="red")
    plt.plot(x, y2, color="red")
    plt.gca().set_aspect(aspect='equal')
    plt.show()

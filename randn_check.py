import matplotlib.pyplot as plt
import numpy as np


samples = np.random.randn(100000)
plt.hist(samples, bins=100, density=True)
plt.title('Randn distribution')
plt.show

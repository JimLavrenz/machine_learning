
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

rng = np.random.RandomState(0)
X, y = make_regression(n_samples=100, n_features=1, random_state=0, noise=14.0,
                       bias=100.0)

plt.plot(X, y, 'b.')
plt.title('Linear Regression')
plt.grid(True)
plt.show()

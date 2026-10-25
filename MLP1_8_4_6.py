import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layer_sizes, activation='sigmoid'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            weight_matrix = np.random.randn(in_dim, out_dim) * 0.1
            bias_vector = np.zeros((1, out_dim))
            self.W.append(weight_matrix)
            self.b.append(bias_vector)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_with_activations(self, X):
        activations = [X]
        A = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A.dot(W) + b
            if i < len(self.W) - 1:
                A = self._sigmoid(Z)
            else:
                A = Z
            activations.append(A)
        return activations

# Инициализируем модель с 4 слоями: 10 входов, 32 нейрон, 16 нейронов и 5 выходов
np.random.seed(42)
model = MLP([10, 32, 16, 5], activation='sigmoid')
X_rand = np.random.randn(1000, 10)

# Соберем активации
acts = model.forward_with_activations(X_rand)

# Отобразим активации каждого слоя
for idx, A in enumerate(acts):
    plt.figure()
    plt.hist(A.flatten(), bins=50)
    plt.title(f'Слой {idx} Распределение активации (часть {A.shape})')
    plt.xlabel('Значение активации')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()
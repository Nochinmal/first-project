import numpy as np
import matplotlib.pyplot as plt

import numpy as np
class MLP:
    def __init__(self, layer_sizes, activation='sigmoid'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.W, self.b = [], []
        # сюда будем складывать Z и A, а также градиенты
        self.Z_list, self.A_list = [], []
        self.dW_list, self.db_list = [], []

        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            self.W.append(np.random.randn(in_dim, out_dim) * 0.1)
            self.b.append(np.zeros((1, out_dim)))

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _dactivate(self, Z):
        # Производная сигмоида через Z
        s = self._sigmoid(Z)
        return s * (1 - s)

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def forward(self, X):
        self.Z_list, self.A_list = [], []
        A = X; self.A_list.append(A)

        for i in range(len(self.W)):
            Z = A.dot(self.W[i]) + self.b[i]
            self.Z_list.append(Z)

            # скрытые слои – sigmoid
            if i < len(self.W) - 1:
                A = self._sigmoid(Z)     
            else:
                # выходной слой – линейный
                A = Z

            self.A_list.append(A)

        return A

    def backward(self, y_true):
        m = y_true.shape[0]

        # δ на выходном слое (линейный + MSE)
        aL = self.A_list[-1]
        delta = 2 * (aL - y_true) / m   # δ^L = 2*(A^L - Y)/m

        # Градиенты для выходного слоя
        a_prev = self.A_list[-2]
        dW = a_prev.T.dot(delta)        # dW^L = A^{L-1}ᵀ · δ^L
        db = np.sum(delta, axis=0, keepdims=True)
        self.dW_list, self.db_list = [dW], [db]

        # 3) Обратный проход по скрытым слоям (sigmoid)
        for l in range(len(self.layer_sizes) - 2, 0, -1):
            z = self.Z_list[l-1]       
            a_prev = self.A_list[l-1]
            W_next = self.W[l]

            # δ^l = (δ^{l+1} · W^{l+1}ᵀ) * σ'(z^l)
            delta = delta.dot(W_next.T) * self._dactivate(z)

            dW = a_prev.T.dot(delta)    # dW^l = A^{l-1}ᵀ · δ^l
            db = np.sum(delta, axis=0, keepdims=True)

            self.dW_list.insert(0, dW)  # вставляем вперёд
            self.db_list.insert(0, db)
            
x = np.array([[1.0, 2.0, -1.0]])
for act in ['relu', 'sigmoid', 'tanh']:
    np.random.seed(0)
    model = MLP([3,2,1], activation=act)
    y_pred = model.forward(x)
    print(f"{act}: y_pred={y_pred.ravel()[0]:.4f}, "
          f"len(Z_list)={len(model.Z_list)}, len(A_list)={len(model.A_list)}")

    # Отобразим значения Z и A
    for i, (Z, A) in enumerate(zip(model.Z_list, model.A_list)):
        print(f"Layer {i}: Z={Z.ravel()}, A={A.ravel()}")
    print()
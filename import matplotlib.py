import numpy as np
import matplotlib.pyplot as plt

# ---------- данные ----------
np.random.seed(0)
X = np.linspace(-2 * np.pi, 2 * np.pi, 300)[:, None]
y = np.sin(X)

# ---------- модель ----------
model = MLP([1, 32, 32, 1], activation='tanh')

# ---------- гиперпараметры ----------
lr = 0.01
epochs = 300
batch_size = 30
loss_history = []

# ---------- цикл обучения ----------
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(len(X))
    X_sh, y_sh = X[perm], y[perm]

    for start in range(0, len(X), batch_size):
        xb = X_sh[start:start + batch_size]
        yb = y_sh[start:start + batch_size]

        model.forward(xb)          # прямой проход
        model.backward(yb)         # обратный проход
        model.update_params(lr)    # обновление параметров

    full_pred = model.forward(X)
    loss_history.append(model.mse(full_pred, y))

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, loss={loss_history[-1]:.4f}")

# ---------- визуализация ----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("MSE по эпохам")
plt.xlabel("Эпоха")
plt.grid(True)

plt.subplot(1, 2, 2)
xx = np.linspace(-2 * np.pi, 2 * np.pi, 200)[:, None]
plt.plot(xx, np.sin(xx), label="sin(x)")
plt.plot(xx, model.predict(xx) if hasattr(model, 'predict') else model.forward(xx),
         label="MLP")
plt.legend()
plt.grid(True)
plt.show()
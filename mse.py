import numpy as np
import matplotlib.pyplot as plt

lr = 0.01
epochs = 300
batch_size = 30
loss_history = []

for epoch in range(1, epochs+1):
    perm = np.random.permutation(len(X))
    X_sh, y_sh = X[perm], y[perm]
    for start in range(0, len(X), batch_size):
        xb = X_sh[start:start+batch_size]
        yb = y_sh[start:start+batch_size]

        # Прямой проход
        preds = model.forward(xb)

        # Обратный проход
        model.backward(yb)

        # Обновление параметров
        model.update_params(lr)

    full_pred = model.forward(X)

    # Считаем потери на данных и добавляем в историю
    loss_history.append(model.mse(full_pred, y))
    
    # Выводим потери каждые 50 эпох
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, loss={loss_history[-1]:.4f}")

# Визуализация
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_history); plt.title("MSE по эпохам"); plt.xlabel("Эпоха"); plt.grid(True)
plt.subplot(1,2,2)
xx = np.linspace(-2*np.pi, 2*np.pi, 200)[:, None]
plt.plot(xx, np.sin(xx), label="sin(x)")
plt.plot(xx, model.forward(xx), label="MLP")
plt.legend(); plt.grid(True)
plt.show()
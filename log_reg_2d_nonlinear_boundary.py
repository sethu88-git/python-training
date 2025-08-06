import numpy as np
import matplotlib.pyplot as plt


def input_generator(r, c, n):
    x = np.hstack((np.random.randn(n, 1),
                  np.random.randn(n, 1)))
    print(f"type(x): {type(x)}, shape: {getattr(x, 'shape', None)}")

    print(f"type(c): {type(c)}, value: {c}")

    z = ((x[:, 0]-c[0])**2).reshape(-1, 1) + ((x[:, 1]-c[1])**2).reshape(-1, 1) - r*r
    y = (sigmoid(z) >= 0.5).astype(int)
    return (x, y)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def bce(y_true, y_pred, w, lam=0):
    eps = 10**-15
    loss = -np.mean(y_true*np.log(np.clip(y_pred, eps, (1-eps))) +
                    (1-y_true)*np.log(1-np.clip(y_pred, eps, (1-eps)))+(lam/2*y_true.shape[0])*(np.linalg.norm(w))**2)
    return loss


def feature_engineer(x):
    x1 = x[:, 0].reshape(-1, 1)
    x2 = x[:, 1].reshape(-1, 1)
    x1_sq = x1 ** 2
    x2_sq = x2 ** 2
    x1_x2 = x1 * x2
    return np.hstack((x1, x2, x1_sq, x2_sq, x1_x2))


def grad_desc(x, y_true):
    alpha = 0.001
    epoch = 10000
    lam = 0.01
    w = np.random.rand(5, 1)
    b = np.random.rand(1, 1)
    error = []
    x_feat = feature_engineer(x)
    for _ in range(epoch):
        y_pred = sigmoid(x_feat @ w+b)
        e = y_pred-y_true
        error.append(bce(y_true, y_pred, w, lam))
        del_w = (x_feat.T@e)/x.shape[0] + lam/x.shape[0] * w
        del_b = np.sum(e)/x.shape[0]
        w += -alpha*del_w
        b += -alpha*del_b
    plot_error(error)
    return (w, b)


def plot_xy(x, y, w, b):
    x1 = x[:, 0]
    x2 = x[:, 1]
    plt.scatter(x1[y.flatten() == 0], x2[y.flatten() == 0], color="red", label="class 1")
    plt.scatter(x1[y.flatten() == 1], x2[y.flatten() == 1], color="green", label="class 2")
    x3 = np.linspace(np.min(x1)*1.1, np.max(x1)*1.1, 100)
    x4 = np.linspace(np.min(x2)*1.1, np.max(x2)*1.1, 100)
    X3, X4 = np.meshgrid(x3, x4)
    z = (w[0] * X3 +
         w[1] * X4 +
         w[2] * X3**2 +
         w[3] * X4**2 +
         w[4] * X3 * X4 +
         b)
    plt.contour(X3, X4, z, levels=0)
    plt.gca().set_aspect(aspect='equal')
    plt.show()


def plot_error(e):
    plt.plot(range(len(e)), e)
    plt.title('Cost function w.r.t iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()


def scale(x, x_t=None):
    x_scaled = (x-np.mean(x, axis=0))/np.std(x, axis=0)
    if x_t is None:
        x_t_scaled = None
    else:
        x_t_scaled = (x_t-np.mean(x, axis=0))/np.std(x, axis=0)
    return (x_scaled, x_t_scaled)


if __name__ == "__main__":
    w, b = [[1.5], [-5.0]], 0.0
    x, y_true = input_generator(1.5, (0, 0), 100)
    w_trained, b_trained = grad_desc(scale(x)[0], y_true)
    x_test, y_test = input_generator(1.5, (0, 0), 10)
    y_pred = sigmoid(feature_engineer(scale(x, x_test)[1])@w_trained + b_trained)
    plot_xy(scale(x)[0], y_true, w_trained.flatten(), b_trained)
    print(np.hstack((y_test, y_pred)))

import numpy as np
import matplotlib.pyplot as plt


def input_generator(w, b):
    ''' 3 unscaled features, 100 samples, input (100,3), weights (3,1), output (100,1)'''
    x = np.hstack((np.random.randint(20, 60, (100, 1)), np.random.randint(
        1, 5, (100, 1)), np.random.randint(1, 150, (100, 1))))
    y = x@w + b + np.random.rand(100, 1)
    return (x, y)


def input_scaling(x, x_t=np.zeros((1, 3))):
    '''normalisation using z-score'''
    x_scaled = (x-np.mean(x, axis=0))/np.std(x, axis=0)
    x_t_scaled = (x_t-np.mean(x, axis=0))/np.std(x, axis=0)
    return (x_scaled, x_t_scaled)


def grad_desc(x, y, epoch=4000):
    costfunc = []
    alpha = 0.001
    lam = 0.05
    w = np.random.rand(3, 1)
    b = np.random.rand(1, 1)
    for i in range(epoch):
        y_pred = x@w + b
        e = y_pred - y
        costfunc.append(np.mean(e*e) + lam/(2*x.shape[0])*np.sum(w**2))
        del_w = 2*(x.T @ e)/x.shape[0] + lam/x.shape[0]*w
        del_b = 2*sum(e)/x.shape[0]
        w -= alpha*del_w
        b -= alpha*del_b
    plot_costfunc(costfunc)
    return (w, b)


def plot_costfunc(e):
    plt.plot(range(len(e)), e)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    w = np.array([[2.8], [-3.2], [0.9]])
    b = [[8.5]]
    x, y = input_generator(w, b)
    x_scaled = input_scaling(x)[0]
    w_trained, b_trained = grad_desc(x_scaled, y)
    x_test = np.hstack((np.random.randint(20, 60, (5, 1)), np.random.randint(
        1, 5, (5, 1)), np.random.randint(1, 150, (5, 1))))
    y_test = x_test@w + b + np.random.rand(5, 1)
    y_pred = input_scaling(x, x_test)[1]@w_trained + b_trained
    print(f"y_pred, y_test:\n{np.hstack((y_pred, y_test))}")

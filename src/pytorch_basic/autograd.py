import torch

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:  # <1>
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():  # <2>
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

def autograd():
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                        3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                        33.9, 21.8, 48.4, 60.4, 68.4])
    t_un = 0.1 * t_u

    params = training_loop(
        n_epochs=5000,
        learning_rate=1e-2,
        params=torch.tensor([1.0, 0.0], requires_grad=True),  # <1>
        t_u=t_un,  # <2>
        t_c=t_c)
    print(params)
    
if __name__ == '__main__':
    autograd()
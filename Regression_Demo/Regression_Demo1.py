import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from Bayes_NN import BayesianNN
from Bayes_Library import minibatch_weight


"""
Demo 1
"""
x = torch.linspace(-2, 2, 500)
y = x.pow(5) - 10 * x.pow(1) + 2 * torch.rand(x.size())
x = torch.unsqueeze(x, dim=1)
y = torch.unsqueeze(y, dim=1)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


def clean_target(x):
    return x.pow(5) - 10 * x.pow(1) + 1


def target(x):
    return x.pow(5) - 10 * x.pow(1) + 2 * torch.rand(x.size())


x_test = torch.linspace(-2, 2, 300)
y_test = target(x_test)

x_test = torch.unsqueeze(x_test, dim=1)
y_test = torch.unsqueeze(y_test, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BayesianNN(in_features=1, hidden_features=200, hidden_num=2, out_features=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()
iteration = 0
epochs = 2000
x, y = x.to(device), y.to(device)

for step in range(epochs):
    iteration = iteration + 1
    pre = model(x)
    loss = model.elbo(
        inputs=x,
        targets=y,
        criterion=criterion,
        n_samples=10,
        w_complexity=1 / x.shape[0]
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iteration % 100 == 0:
        print("Iteration {}: Loss: {:.4f}".format(iteration, loss))

count = 30
pred = model.predict(x=x_test.to(device), samples=count, statistic_flag=False)
pred = pred.cpu().numpy()
plt.figure(figsize=(10,8), dpi=80)
plt.plot(x_test.data.numpy(),y_test.data.numpy(),'.',color='darkorange',markersize=4,label='Test set')
plt.plot(x_test.data.numpy(),clean_target(x_test).data.numpy(),color='green',markersize=4,label='Target function')
for i in range(count):
    pred_ = pred[i]
    plt.plot(x_test.data.numpy(), pred_, "b-", alpha=1.0 / 15, label="Particle estimate")
plt.show()

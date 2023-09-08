import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from Bayes_NN import BayesianNN
from Bayes_Library import minibatch_weight

"""
Demo 2
"""


def secret_function(x, noise=0.0):
    return x.sin() + noise * torch.randn_like(x)


X = 8 * torch.rand(20, 1) - 4
Y = secret_function(X, noise=1e-1)

x = torch.linspace(-4, 4, steps=10).reshape(-1, 1)
y = secret_function(x)

x_test = torch.linspace(-4, 4, 100).reshape(-1, 1)
y_test = secret_function(x_test, 1e-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BayesianNN(in_features=1, hidden_features=100, hidden_num=1, out_features=1)
model = model.to(device)

ds_train = TensorDataset(x, y)
dataloader_train = DataLoader(ds_train, batch_size=10, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()
epochs = 2000
iteration = 0
for epoch in range(epochs):
    train_loss = 0.0
    for batch_idx, (data, labels) in enumerate(dataloader_train):
        data, labels = data.to(device), labels.to(device)
        # pi_weight = minibatch_weight(batch_idx=batch_idx, num_batches=1)
        optimizer.zero_grad()
        loss = model.elbo(
            inputs=data,
            targets=labels,
            criterion=criterion,
            n_samples=10,
            w_complexity=1 / x.shape[0]
        )

        loss.backward()
        optimizer.step()
    iteration += 1

    if iteration % 100 == 0:
        print("Iteration {}: Loss: {:.4f}".format(iteration, loss))

plt.figure(figsize=(12, 6))
count = 30
pred = model.predict(x_test.to(device), count, statistic_flag=False)
pred = pred.cpu().numpy()
plt.plot(X.numpy(), Y.numpy(), "rs", ms=4, label="Sampled points")
plt.plot(x, y, "r--", label="Ground truth")
for i in range(count):
    pred_ = pred[i]
    plt.plot(x_test.cpu().numpy(), pred_, "b-", alpha=1.0 / 15, label="Particle estimate")
plt.show()

import torch
import torch.nn as nn
from Bayes_Library import variational_approximation, BayesianLinear


@variational_approximation
class BayesianNN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_num, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = nn.ReLU()

        hidden_layers = []
        for idx in range(hidden_num):
            if idx == 0:
                hidden_layers.append(BayesianLinear(in_features, hidden_features))
                hidden_layers.append(self.activation)
            else:
                hidden_layers.append(BayesianLinear(hidden_features, hidden_features))
                hidden_layers.append(self.activation)
        self.hidden = nn.Sequential(*hidden_layers)
        self.exit = BayesianLinear(hidden_features, out_features)

    def forward(self, x):
        x = self.hidden(x)
        x = self.exit(x)
        return x

    def predict(self, x, samples=1, statistic_flag=False):
        pred = torch.zeros(samples, x.shape[0], self.out_features)
        for i in range(samples):
            with torch.no_grad():
                pred[i] = self.forward(x)

        mean = pred.mean(axis=0)
        std = pred.std(axis=0)

        if statistic_flag:
            return mean, std
        else:
            return pred


if __name__ == '__main__':

    bayesian_model = BayesianNN(in_features=2, hidden_features=30, hidden_num=2, out_features=1)

    input_data = torch.tensor([[0.5, 0.3], [0.2, 0.7]])

    mean_predictions, variance_predictions = bayesian_model.predict(input_data, samples=100, statistic_flag=True)
    predictions = bayesian_model.predict(input_data, samples=10, statistic_flag=False)
    print("Mean Predictions:", mean_predictions)
    print("Variance Predictions:", variance_predictions)
    print('Predictions:', predictions)

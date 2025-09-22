import torch

# Feedforward neural network with one hidden layer of relu activation, and sigmoid output activation.
class FeedForwardNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(FeedForwardNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
class FeedForwardNN4Layered(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size_3, output_size, dropout_rate=0.05):
        super(FeedForwardNN3Layered, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.fc3 = torch.nn.Linear(hidden_size2, output_size)
        self.dropout3 = torch.nn.Dropout(dropout_rate)
        self.fc4 = torch.nn.Linear(hidden_size_3, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out
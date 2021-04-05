class MLP_2layer(nn.Module):
    def __init__(self, CFG, x_train, targets):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(CFG.hidden_size*2, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, 22))

    def forward(self, x):
        x = self.mlp(x)
        return x


class LSTM_2layer(nn.Module):
    def __init__(self, CFG, x_train, targets):
        super().__init__()
        self.rnn = nn.LSTM(22, CFG.hidden_size, batch_first = True)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(nn.Linear(CFG.hidden_size*2, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, 22))

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
      
  class GRU_2layer(nn.Module):
    def __init__(self, CFG, x_train, targets):
        super().__init__()
        self.rnn = nn.GRU(22, CFG.hidden_size, batch_first = True)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(nn.Linear(CFG.hidden_size*2, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, 22))

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
      
  class LSTM_GRU_2layer(nn.Module):
    def __init__(self, CFG, x_train, targets):
        super().__init__()
        self.rnn1 = nn.LSTM(22, 22, batch_first = True)
        self.rnn2 = nn.GRU(22, CFG.hidden_size, batch_first = True)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(nn.Linear(CFG.hidden_size*2, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.ReLU(),
                          nn.Linear(CFG.hidden_size, 22))

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

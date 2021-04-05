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


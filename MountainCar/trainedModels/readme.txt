MountainCar:

The file 'MountainCar_False_1_perfect.dat' contains the neural network weights.
Net is a Duel Linear without Noise where
1- in_features = 2
2- n_actions = 3


class DuelDQN(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()

        self.input = nn.Linear(in_features, 256)
        self.fc_adv = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.input(x.float()))
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return (val + (adv - adv.mean(dim=-1, keepdim=True)))
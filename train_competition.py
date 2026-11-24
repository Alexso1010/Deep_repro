"""
Competition script: forecast the next 96 hours of oil temperature (OT) on ETTh1.

Propose your own models in the MODELS section below and select them with the hydra configs:
    uv run train_competition.py participant=<your_name> model=competition_mlp

Model contract:
  - __init__ receives your hyperparameters from conf/model/<model>.yaml, plus
    `lookback`, `n_channels`, `horizon` and `target_idx`, injected by the harness.
  - forward maps X of shape (batch, lookback, n_channels) to (batch, horizon),
    the standardized OT forecast. X holds the 7 standardized ETTh1 channels, OT is X[:, :, target_idx].

Training, evaluation and logging live in competition/harness.py: do not edit it.
"""

import hydra
from omegaconf import DictConfig
from torch import nn

from competition.harness import run_competition

###############################################################################
#                                  MODELS                                     #
###############################################################################


class TSMLP(nn.Module):
    """
    An MLP on the flattened history of all channels.
    """

    def __init__(self, lookback, n_channels, horizon, target_idx, hidden_dim, n_layers, dropout):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(lookback * n_channels, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, horizon)

    def forward(self, X):
        out = self.mlp(X)
        y_hat = self.linear(out)
        return y_hat


###############################################################################
#                               MAIN FUNCTION                                 #
###############################################################################


@hydra.main(config_path="conf", config_name="competition", version_base="1.3")
def main(cfg: DictConfig):
    return run_competition(cfg)


if __name__ == "__main__":
    main()

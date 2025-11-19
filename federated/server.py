import os
import torch
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.grid.grid import Grid   # ← correct for Flower 1.23
from centralized.model import TimeSeriesLSTM

app = ServerApp()

@app.main()
def main(grid:Grid, context: Context):

    # ---- Load config ----
    cfg = context.run_config
    num_rounds   = cfg.get("num-server-rounds", 3)
    fraction_fit = cfg.get("fraction-fit", 1.0)
    lr           = cfg.get("lr", 0.001)
    local_epochs = cfg.get("num-local-epochs", 1)
    batch_size   = cfg.get("batch-size", 32)
    
    # ---- Initialize model ----
    model = TimeSeriesLSTM()
    ndarrays = [p.detach().cpu().numpy() for p in model.state_dict().values()]
    initial_params = ndarrays_to_parameters(ndarrays)

    # ---- Config passed to clients ----
    def fit_config_fn(rnd: int):
        return {
            "lr": lr,
            "num-local-epochs": local_epochs,
            "batch-size": batch_size,
        }

    # ---- Strategy ----
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        initial_parameters=initial_params,
        on_fit_config_fn=fit_config_fn,
    )

    # ---- Return to Flower runtime ----
    return strategy, {"num_rounds": num_rounds}

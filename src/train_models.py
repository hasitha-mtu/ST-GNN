import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_per_node_gru_model import train as train_gru
from train_per_node_lstm_model import train as train_lstm

from train_st_gnn_flood_model import train as train_st_gnn
from train_st_gnn_flood_model_sar import train as train_st_gnn_sar
from train_st_gnn_dyn_edge import train as train_st_gnn_dyn_edge
from train_st_gnn_hand_edge import train as train_st_gnn_hand_edge

from train_dfc_gnn import train as train_dfc_gnn

from utils.common_utils import seed_everything
from utils.config import load_config
from utils.logger import get_logger

from pathlib import Path

MAX_EPOCHS = 1

if __name__ == "__main__":
    # seeds = [42, 123, 456]
    seeds = [42]
    t_outs = [4, 12, 16]
    t_in = 32
    max_epochs = MAX_EPOCHS
    config_path = r"C:\Users\AdikariAdikari\PycharmProjects\ST-GNN\config\config.yaml"
    config = load_config(Path(config_path))
    logger = get_logger(config["logging"]["train"])
    for seed in seeds:
        seed_everything(seed)
        for t_out in t_outs:
            train_gru(logger, seed, t_in, t_out, max_epochs)
            train_lstm(logger, seed, t_in, t_out, max_epochs)

            train_st_gnn(logger, seed, t_in, t_out, max_epochs)
            train_st_gnn_sar(logger, seed, t_in, t_out, max_epochs)
            train_st_gnn_dyn_edge(logger, seed, t_in, t_out, max_epochs)
            train_st_gnn_hand_edge(logger, seed, t_in, t_out, max_epochs)

            train_dfc_gnn(logger, seed, t_in, t_out, max_epochs)

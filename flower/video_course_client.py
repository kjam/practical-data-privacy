import argparse
import json
from typing import Dict, List, Tuple, Optional

import flwr as fl

from flwr.common import (
    Scalar,
    parameters_to_ndarrays,
)

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X: Dict):
        self.list = X['list_items']
        self.stores = X['store_location_preferences']

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        truncated_list = [np.asarray(
            [x['item_id'], x['quantity']] + self.stores[:2]) for x in self.list]
        return (
            truncated_list,
            len(truncated_list),
            {},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Optional[Tuple[float, int, Dict]]:
        item_ids = [x['item_id'] for x in self.list]
        forecast_df = pd.DataFrame(parameters,
            columns=['store_id', 'item_id', 'forecast_quantity'])
        personal_forecast = forecast_df.loc[ \
            (forecast_df['store_id'].isin(self.stores)) & \
            (forecast_df['item_id'].isin(item_ids)) ]

        total_score_per_store = personal_forecast.groupby('store_id')['forecast_quantity'].sum()
        print('personal forecast score', total_score_per_store)

        # we don't actually need this for our first goal, but I
        # Challenge: Is there a creative use for this metric?
        # Maybe a customer satisfaction score?? :)
        return 0.1, 1, {"foo": 0.1}




if __name__ == "__main__":
    N_CLIENTS = 5

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the partition id of artificially partitioned datasets.",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    # Load the shoping list data
    with open('data/lists.json') as f:
        data = json.load(f)
    list = data[partition_id]
        # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(list).to_client(),
    )

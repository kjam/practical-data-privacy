from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import pandas as pd
import itertools
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class FedAnalytics(Strategy):
    def initialize_parameters(
        self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=5, min_num_clients=5)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Get results from fit
        # Convert results

        values_aggregated = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        lists_df = pd.DataFrame(itertools.chain.from_iterable(values_aggregated),
            columns=['item_id', 'quantity', 'store_id', 'extra_store'])

        # some reshaping to do easier math
        extended_df = lists_df[['item_id', 'quantity', 'extra_store']].copy()
        extended_df['store_id'] = extended_df['extra_store']
        extended_df = extended_df.drop(['extra_store'], axis=1)
        lists_df = lists_df.drop(['extra_store'], axis=1)
        lists_df = pd.concat([lists_df, extended_df])

        # starting math bit
        lists_df['quantity'] = lists_df['quantity'] * -1 # lists remove quanities
        inventory_df = pd.read_json('data/inventory.json')
        combined_df = pd.concat([inventory_df.drop(['category', 'item'], axis=1), lists_df])
        forecast_df = combined_df.groupby(['store_id', 'item_id'], as_index=False).sum()

        return ndarrays_to_parameters(forecast_df.to_numpy()), {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        agg_hist = [arr for arr in parameters_to_ndarrays(parameters)]
        return 0, {"Aggregated forecast": agg_hist}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        eval_ins = EvaluateIns(parameters, {})

        # Challenge: what if you just sent back the parts of the inventory
        # with low medium high so that you aren't revealing proprietary data?

        # Extra challenge: what if you only sent back data to clients that matched
        # their store list? 

        clients = client_manager.sample(num_clients=5, min_num_clients=5)
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return 0.1, {'test': 1}


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=FedAnalytics(),
)

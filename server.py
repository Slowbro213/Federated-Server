import flwr as fl

# Define strategy (e.g., Federated Averaging)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of clients for training
    fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
    min_fit_clients=3,  # Minimum number of clients to train
    min_evaluate_clients=3,  # Minimum number of clients to evaluate
    min_available_clients=3,  # Minimum number of clients available
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
class Config:

    dataset = "HCPD"

    num_nodes = 116

    in_dim = 116
    hidden_dim = 128

    batch_size = 16
    lr = 1e-3

    epochs = 200

    lambda_cm = 0.1
    lambda_sg = 0.01

    task = "regression"
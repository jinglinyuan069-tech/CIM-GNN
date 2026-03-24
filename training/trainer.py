import torch

from training.loss import total_loss


class Trainer:

    def __init__(self, model, config):

        self.model = model
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=1e-5
        )


    def train_epoch(self, loader):

        self.model.train()

        total_loss_value = 0

        for fc_data, sc_data, y in loader:

            fc_data = fc_data.to(self.device)
            sc_data = sc_data.to(self.device)
            y = y.to(self.device)

            pred, h_fc, h_sc, edge_mask, alpha = self.model(
                fc_data,
                sc_data
            )

            loss, losses = total_loss(
                pred,
                y,
                h_fc,
                h_sc,
                edge_mask,
                self.config
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss_value += loss.item()

        avg_loss = total_loss_value / len(loader)

        return avg_loss


    def validate(self, loader):

        self.model.eval()

        total_loss_value = 0

        with torch.no_grad():

            for fc_data, sc_data, y in loader:

                fc_data = fc_data.to(self.device)
                sc_data = sc_data.to(self.device)
                y = y.to(self.device)

                pred, h_fc, h_sc, edge_mask, alpha = self.model(
                    fc_data,
                    sc_data
                )

                loss, losses = total_loss(
                    pred,
                    y,
                    h_fc,
                    h_sc,
                    edge_mask,
                    self.config
                )

                total_loss_value += loss.item()

        avg_loss = total_loss_value / len(loader)

        return avg_loss


    def train(self, train_loader, val_loader=None, epochs=200):

        history = []

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)

            if val_loader is not None:

                val_loss = self.validate(val_loader)

                print(
                    f"Epoch {epoch:03d} | "
                    f"Train {train_loss:.4f} | "
                    f"Val {val_loss:.4f}"
                )

                history.append((train_loss, val_loss))

            else:

                print(
                    f"Epoch {epoch:03d} | "
                    f"Train {train_loss:.4f}"
                )

                history.append((train_loss, None))

        return history
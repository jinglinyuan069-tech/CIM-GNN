import os
import sys
import time


class Logger:

    def __init__(self, log_dir="logs", name="experiment"):

        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.log_path = os.path.join(
            log_dir,
            f"{name}_{timestamp}.log"
        )

        self.file = open(self.log_path, "a")

    def write(self, message):

        message = str(message)

        sys.stdout.write(message + "\n")
        sys.stdout.flush()

        self.file.write(message + "\n")
        self.file.flush()

    def log(self, message):

        self.write(message)

    def close(self):

        self.file.close()


def log_config(logger, config):

    logger.log("===== Config =====")

    for key, value in vars(config).items():

        logger.log(f"{key}: {value}")

    logger.log("==================")


def log_metrics(logger, epoch, metrics):

    msg = f"Epoch {epoch} | "

    for k, v in metrics.items():

        msg += f"{k}: {v:.4f} "

    logger.log(msg)
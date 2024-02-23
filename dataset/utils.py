import logging
logger = logging.getLogger(__name__)


def setup_logger(log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)      
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(handler)



import pandas as pd

class CSV_RW:
    def __init__(self, path) -> None:
        self.path = path

    def check_row_exists(self, row):
        df = pd.read_csv(self.path, delimiter=",")
        row_exists = (df['model_name'] == row['model_name']) & \
                (df['model_dtype'] == row['model_dtype']) & \
                (df['batch_size'] == row['batch_size']) & \
                (df['drop_rate'] == row['drop_rate']) & \
                (df['opt'] == row['opt']) & \
                (df['lr'] == row['lr']) & \
                (df['weight_decay'] == row['weight_decay'])

        if row_exists.any():
            return True
        else:
            return False
        
    def write_csv(self, row):
        df = pd.DataFrame(row, index=[0])
        df.to_csv(self.path, index=False, mode='a', header=False)


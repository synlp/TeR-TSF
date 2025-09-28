import logging
import os
from utils.tools import print_with_timestamp

class Logger:
    def __init__(self, log_path, log_file_name="app.log", log_level=logging.INFO):
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_handler = logging.FileHandler(os.path.join(log_path,log_file_name))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


    def debug(self, message):
        print_with_timestamp(message)
        self.logger.debug(message)

    def info(self, message):
        print_with_timestamp(message)
        self.logger.info(message)

    def warning(self, message):
        print_with_timestamp(message)
        self.logger.warning(message)

    def error(self, message):
        print_with_timestamp(message)
        self.logger.error(message)

    def critical(self, message):
        print_with_timestamp(message)
        self.logger.critical(message)

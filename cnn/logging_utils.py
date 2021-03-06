import logging


def init_logging(log_file, log_level):
    logging.getLogger().setLevel(log_level)
    log_formatter = logging.Formatter('%(asctime)s %(message)s')
    root_logger = logging.getLogger()
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info('Logging to %s', log_file)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

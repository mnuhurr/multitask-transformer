
import logging


def init_log(name, filename=None, level='info', stream=True):
    logger = logging.getLogger(name)

    if type(level) == str:
        level = level.upper()

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    if filename is not None:
        filehandler = logging.FileHandler(filename)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    if stream:
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)

    logger.propagate = False

    return logger



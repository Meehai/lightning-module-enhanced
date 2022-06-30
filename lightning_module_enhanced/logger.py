"""Python logger settings."""

import os
import logging

KEY = "LIGHTNING_MODULE_ENHANCED"
ENV_KEY = f"{KEY}_LOGLEVEL"
env_var = int(os.environ[ENV_KEY]) if ENV_KEY in os.environ else 2

# Usage: loglevel=0 (none), loglevel=1 (info), loglevel=2 (debug), loglevel=3 (debug verbose)
logging.DEBUG2 = 3
loglevel = {0: logging.NOTSET, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG2}[env_var]
logging.addLevelName(logging.DEBUG2, "DEBUG-VERBOSE")

class CustomFormatter(logging.Formatter):
    """Custom formatting for logger."""

    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    pre = "[%(asctime)s-%(name)s-%(levelname)s]"
    post = "(%(filename)s:%(funcName)s:%(lineno)d)"

    # Example [TIME:LEVEL:NAME] Message [FILE:FUNC:LINE]
    FORMATS = {
        logging.DEBUG: f"{cyan}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.DEBUG2: f"{cyan}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.INFO: f"{green}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.WARNING: f"{yellow}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.ERROR: f"{red}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.CRITICAL: f"{bold_red}{pre}{reset} %(message)s {yellow}{post}{reset}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.formatTime = self.formatTime
        return formatter.format(record)

    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, "%Y-%m-%d %H:%M:%S")

# instantiate logger and set log level
logger = logging.getLogger(KEY)
logger.setLevel(loglevel)
logger.debug2 = lambda msg: logger.log(logging.DEBUG2, msg)

# add custom formatter to logger
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)
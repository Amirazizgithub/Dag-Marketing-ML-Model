import sys
import pytz
import logging
from datetime import datetime

# 1. Indian Timezone Configuration
IST = pytz.timezone("Asia/Kolkata")

# --- NEW: Register SUCCESS level ---
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)


# Add the success method to the Logger class
logging.Logger.success = success
# -----------------------------------


class ISTFormatter(logging.Formatter):
    """Custom formatter to force IST timestamps in logs"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def setup_gke_logging():
    # 2. Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 3. Create a StreamHandler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # 4. Set the IST Formatter
    # 4. Set the IST Formatter
    # formatter = ISTFormatter(
    #     "%(asctime)s | %(levelname)-8s | %(filename)s:line-%(lineno)d - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    formatter = ISTFormatter(
        "%(levelname)-8s | %(filename)s:line-%(lineno)d - %(message)s",
    )
    handler.setFormatter(formatter)

    # 5. Avoid duplicate logs if setup is called twice
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# Usage
log = setup_gke_logging()

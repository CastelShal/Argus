import os
import logging
from logging.handlers import TimedRotatingFileHandler
LOG_DIR = "logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setLoggers():
    camHandler = TimedRotatingFileHandler(
        "logs/daily.log",
        when="midnight",
        interval=1
    )

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    camHandler.setFormatter(formatter)

    camLogger = logging.getLogger("CameraLogger")
    camLogger.addHandler(camHandler)
    camLogger.setLevel(logging.INFO)

    adminLogger = logging.getLogger("AdminLogger")
    adminHandler = logging.FileHandler("logs/admin.log")

    adminHandler.setFormatter(formatter)
    adminLogger.addHandler(adminHandler)
    adminLogger.setLevel(logging.DEBUG)

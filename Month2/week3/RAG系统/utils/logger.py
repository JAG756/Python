# utils/logger.py
import logging
import os
from config import LOG_LEVEL, LOG_FILE, LOG_FORMAT

# 获取日志等级
level_dict = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR
}

# 配置日志
logging.basicConfig(
    level=level_dict[LOG_LEVEL],
    format=LOG_FORMAT,
    handlers=[
        # 控制台输出
        logging.StreamHandler(),
        # 文件输出（自动保存日志）
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)

# 全局日志对象（整个项目共用）
logger = logging.getLogger("RAG_System")
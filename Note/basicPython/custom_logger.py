import logging
import os

class Logger(object):
    def __init__(self, log_name, log_level, log_path):
        self.log_name = log_name
        self.log_level = log_level
        self.log_path = log_path

        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=self.log_level)

        # 统一格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 配置 FileHandler
        file_handler = logging.FileHandler(self.log_path, "w")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)

        # 配置 StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.log_level)
        stream_handler.setFormatter(formatter)

        # 添加 Handler 到 logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

####################################################################
if __name__ == "__main__":
    logger = Logger('Class-Logger', logging.INFO, './logtest.log')
    mylogger = logger.init_logger()

    mylogger.info("测试info")
    mylogger.debug("测试debug")
    mylogger.warning("测试warning")
    mylogger.error("测试error")
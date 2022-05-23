import logging

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger.info("测试info")
logger.debug("测试debug")
logger.warning("测试warning")
logger.error("测试error")
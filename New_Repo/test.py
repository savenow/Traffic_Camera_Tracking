import logging

logging.basicConfig(format='[%(levelname)s: %(message)s]', level=logging.DEBUG)

logging.debug("This is a debug message")
logging.critical("This is a critical mistake")
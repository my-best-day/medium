import logging
import torch


logger = logging.getLogger(__name__)


def _main():
    logging.info("Hello %s!", "World")
    print("Hello World!")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main()

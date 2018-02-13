"""
decorators.py

Author: Tobias Seydewitz
Date: 20.09.17
Mail: tobi.seyde@gmail.com

Description:
"""
import time
from functools import wraps


def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        A decorator for benchmark methods. It prints the duration
        the provided function needed to stdout.
        :param args: func arguments
        :param kwargs: func key arguments
        :return: func result
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        print('{}: {}'.format(func.__name__, end))
        return result
    return wrapper


if __name__ == '__main__':
    pass

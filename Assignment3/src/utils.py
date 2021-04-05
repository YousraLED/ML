from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        time_start = time()
        result = f(*args, **kw)
        time_end = time()
        function_name = f.__name__
        duration = round(time_end - time_start, 5)
        print(f'func: {function_name} took: {duration} sec')

        return result
    return wrap
env = {}

def get_env():
    global env
    return env

def manual_seed(seed):
    env = get_env()
    env['manual_seed'] = seed

def print_precision(precision):
    env = get_env()
    env['print_precision'] = precision
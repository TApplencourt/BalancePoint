from functools import lru_cache

def lazy_property(f):
    return property(lru_cache()(f))

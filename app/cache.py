import json
from .extensions import redis_client


# Get a cached set from Redis
def cache_get(key):
    value = redis_client.get(key)
    return json.loads(value) if value else None


# Save information to the Redis cache
def cache_set(key, data, ttl=3600):
    redis_client.set(key, json.dumps(data), ex=ttl)

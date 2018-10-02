#!/usr/bin/env python3
import time
from core.model_factory import get_agent, WEIGHTS, LOCK
import os.path
import redis
import core.redis_credentials as redis_cred
from filelock import FileLock

dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = "{}/{}".format(dir_path, WEIGHTS)
lock_path = "{}/{}".format(dir_path, LOCK)






def main():
    lock = FileLock(lock_path)
    agent = get_agent()

    optimize_queue = redis.StrictRedis(
        host=redis_cred.host,
        port=redis_cred.port,
        password=redis_cred.password)

    channel = optimize_queue.pubsub()
    channel.subscribe(redis_cred.optimizer_subscription)

    while True:
        message = channel.get_message()
        if message and message['data'] == b'optimize':
            with lock:
                os.chmod(lock_path, 0o777)
                if os.path.isfile(weights_path) and os.access(weights_path, os.W_OK & os.R_OK):
                    agent.load_weights(weights_path)
                    metrics = agent.optimize()
                    print("metrics", metrics)
                else:
                    agent.save_weights(weights_path)

                agent.save_weights(weights_path, overwrite=True)
                os.chmod(weights_path, 0o777)


if __name__ == "__main__":
    main()

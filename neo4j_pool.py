"""
neo4j_pool.py — One shared Neo4j driver per (uri, user, password).

Creating a driver means a new connection + handshake + auth. The analysers,
schema explorer, degraders and ML pipeline used to create one per call;
they now all borrow the same driver from here (its close() is a no-op).
"""
import atexit
import threading

from neo4j import GraphDatabase

_drivers: dict = {}
_lock = threading.Lock()


def get_driver(uri: str, user: str, password: str):
    key = (uri, user, password)
    with _lock:
        driver = _drivers.get(key)
        if driver is None:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            _drivers[key] = driver
        return driver


def close_all():
    with _lock:
        for driver in _drivers.values():
            try:
                driver.close()
            except Exception:
                pass
        _drivers.clear()


atexit.register(close_all)

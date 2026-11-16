"""
experiment_jobs.py — Graph ML experiments run in a background thread.

Streamlit stops the running script on every click, so an experiment run inside
the page died as soon as the user touched the app. Jobs live here instead
(module state survives reruns), one per project; the page only polls them.
"""
import atexit
import threading

_jobs: dict = {}
_lock = threading.Lock()


class Job:
    def __init__(self, key, uri, db, meta, target):
        self.key      = key
        self.uri      = uri
        self.db       = db
        self.meta     = meta          # what the page needs to display the results
        self.log_lines: list[str] = []
        self.progress = 0.0
        self.status   = "running"     # running | done | cancelled | failed
        self.results  = None
        self.cancel   = threading.Event()
        self._target  = target
        self.thread   = threading.Thread(target=self._run, daemon=True,
                                         name=f"ml-job-{db}")

    def log(self, msg: str):
        self.log_lines.append(msg)

    def _run(self):
        try:
            self.results = self._target(self)
            self.status = "cancelled" if self.cancel.is_set() else "done"
        except Exception as e:
            self.log(f"✗ Experiment failed: {e}")
            self.status = "failed"


def start(key, uri, db, meta, target) -> Job:
    """Start target(job) in a thread. Refuses a second running job on the same database."""
    with _lock:
        busy = busy_db(uri, db)
        if busy is not None:
            return busy
        job = Job(key, uri, db, meta, target)
        _jobs[key] = job
    job.thread.start()
    return job


def get(key):
    return _jobs.get(key)


def pop(key):
    with _lock:
        return _jobs.pop(key, None)


def busy_db(uri, db):
    """The running job that is degrading this database, if any."""
    for job in list(_jobs.values()):
        if job.status == "running" and job.uri == uri and job.db == db:
            return job
    return None


@atexit.register
def _cancel_all():
    # On server shutdown, let running jobs stop and restore the graph.
    for job in list(_jobs.values()):
        if job.status == "running":
            job.cancel.set()
    for job in list(_jobs.values()):
        job.thread.join(timeout=300)

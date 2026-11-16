"""
project_manager.py — DQGraph project persistence.

Each project is a folder under ~/graphlens_projects/<safe_name>/
  project.json   — metadata + accumulated quality metrics
  ml_cache/      — one JSON file per ML experiment (keyed by config hash)
"""
import json, os, hashlib
from datetime import datetime

PROJECTS_ROOT = os.path.expanduser("~/graphlens_projects")

METRIC_TO_DIMENSION = {
    "property_completeness":  "Completeness",
    "edge_coverage":          "Completeness",
    "type_completeness":      "Completeness",
    "property_conformity":    "Conformity",
    "conditional_conformity": "Conformity",
    "type_conformity":        "Conformity",
    "type_age":               "Freshness",
    "property_age":           "Freshness",
    "update_frequency":       "Freshness",
    "exact_duplicate_rate":   "Uniqueness",
    "partial_duplicate_rate": "Uniqueness",
    "uniqueness_score":       "Uniqueness",
    "property_accuracy":      "Accuracy",
}


def _safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in " _-") else "_" for c in name).strip()


class Project:
    def __init__(self, name: str, uri: str, neo_user: str):
        self.name       = name
        self.uri        = uri
        self.neo_user   = neo_user
        self.created_at = datetime.now().isoformat()
        self.quality_results: list = []

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def dir(self) -> str:
        return os.path.join(PROJECTS_ROOT, _safe_name(self.name))

    @property
    def path(self) -> str:
        return os.path.join(self.dir, "project.json")

    @property
    def ml_cache_dir(self) -> str:
        return os.path.join(self.dir, "ml_cache")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.ml_cache_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "name":            self.name,
                "uri":             self.uri,
                "neo_user":        self.neo_user,
                "created_at":      self.created_at,
                "quality_results": self.quality_results,
                "version":         1,
            }, f, indent=2, default=str)

    # ── Quality results ───────────────────────────────────────────────────────

    def _result_key(self, entry: dict) -> str:
        """Unique key for deduplication: same metric + same db + same params."""
        parts = [entry.get("metric_id", ""), entry.get("db", ""),
                 json.dumps(entry.get("params", {}), sort_keys=True)]
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def upsert_quality_result(self, entry: dict):
        """Add or replace a quality result (same metric+db+params → replace)."""
        key = self._result_key(entry)
        entry["_key"] = key
        entry["computed_at"] = datetime.now().isoformat()
        for i, r in enumerate(self.quality_results):
            if r.get("_key") == key:
                self.quality_results[i] = entry
                self.save()
                return
        self.quality_results.append(entry)
        self.save()

    def clear_quality_results(self, db: str | None = None):
        """Remove the quality results of `db` (all of them if None) and persist."""
        self.quality_results = ([] if db is None else
                                [r for r in self.quality_results if r.get("db") != db])
        self.save()

    def results_by_dimension(self, db: str | None = None) -> dict:
        """Returns {dimension: [entries]} ordered by dimension, only for `db`
        when given."""
        out = {}
        for r in self.quality_results:
            if db is not None and r.get("db") != db:
                continue
            dim = METRIC_TO_DIMENSION.get(r.get("metric_id", ""), "Other")
            out.setdefault(dim, []).append(r)
        return out

    # ── ML cache ──────────────────────────────────────────────────────────────

    def _cache_key(self, config: dict) -> str:
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def load_ml_cache(self, config: dict):
        path = os.path.join(self.ml_cache_dir, f"{self._cache_key(config)}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_ml_cache(self, config: dict, results: dict):
        os.makedirs(self.ml_cache_dir, exist_ok=True)
        path = os.path.join(self.ml_cache_dir, f"{self._cache_key(config)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "config":      config,
                "results":     results,
                "computed_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    def delete_ml_cache(self, config: dict):
        path = os.path.join(self.ml_cache_dir, f"{self._cache_key(config)}.json")
        if os.path.exists(path):
            os.remove(path)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, project_path: str) -> "Project":
        with open(project_path, encoding="utf-8") as f:
            data = json.load(f)
        p = cls(data["name"], data.get("uri", ""), data.get("neo_user", ""))
        p.created_at      = data.get("created_at", p.created_at)
        p.quality_results = data.get("quality_results", [])
        return p

    @classmethod
    def list_all(cls) -> list:
        """Return [{name, path, created_at, uri, n_quality}] newest first."""
        if not os.path.isdir(PROJECTS_ROOT):
            return []
        result = []
        for d in os.listdir(PROJECTS_ROOT):
            proj_path = os.path.join(PROJECTS_ROOT, d, "project.json")
            if not os.path.exists(proj_path):
                continue
            try:
                with open(proj_path, encoding="utf-8") as f:
                    data = json.load(f)
                result.append({
                    "name":      data.get("name", d),
                    "path":      proj_path,
                    "created_at": data.get("created_at", ""),
                    "uri":       data.get("uri", ""),
                    "n_quality": len(data.get("quality_results", [])),
                })
            except Exception:
                pass
        return sorted(result, key=lambda x: x["created_at"], reverse=True)

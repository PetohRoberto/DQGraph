"""
community_detection.py
----------------------
Community Detection via Neo4j Graph Data Science (GDS).

Uses Louvain algorithm to detect communities and compute quality metrics:
  - Modularity      : how well the graph is partitioned
  - Coverage        : proportion of nodes in non-singleton communities
  - Conductance     : average ratio of external edges per community (lower = better)
"""

from __future__ import annotations
from neo4j import GraphDatabase
import time


class CommunityDetectionPipeline:

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def _run(self, query: str, params: dict = None) -> list[dict]:
        with self.driver.session(database=self.database) as s:
            return s.run(query, **(params or {})).data()

    def _one(self, query: str, params: dict = None) -> dict:
        with self.driver.session(database=self.database) as s:
            r = s.run(query, **(params or {})).single()
            return dict(r) if r else {}

    def _drop(self, name: str):
        try:
            self._run(f"CALL gds.graph.drop('{name}', false)")
        except Exception:
            pass

    def run_full_pipeline(
        self,
        node_label: str,
        rel_type: str,
        progress_callback=None,
    ) -> dict:
        """
        Run community detection with Louvain and compute metrics.

        Parameters
        ----------
        node_label  : label of the nodes
        rel_type    : relationship type

        Returns
        -------
        dict with: success, metrics (Modularity, Coverage, Conductance), elapsed
        """
        ts  = time.time()
        uid = str(int(ts))[-6:]
        g   = f"cd_graph_{uid}"

        def log(msg):
            if progress_callback:
                progress_callback(msg)

        try:
            # 1. Project
            log("  Projecting graph for community detection...")
            self._drop(g)
            proj = self._one(
                f"CALL gds.graph.project($g, "
                f"'{node_label}', "
                f"{{{rel_type}: {{orientation: 'UNDIRECTED'}}}}) "
                f"YIELD graphName, nodeCount, relationshipCount",
                {"g": g},
            )
            node_count = proj.get("nodeCount", 0)
            rel_count  = proj.get("relationshipCount", 0)

            if node_count == 0:
                log("  ✗ No nodes in projection")
                return {"success": False, "metrics": {}, "error": "Empty graph",
                        "elapsed": round(time.time() - ts, 1)}

            # 2. Run Louvain stats for modularity
            log("  Running Louvain algorithm (stats)...")
            stats = self._one(
                f"CALL gds.louvain.stats('{g}') "
                f"YIELD modularity, communityCount, communityDistribution"
            )
            modularity      = stats.get("modularity")
            community_count = stats.get("communityCount", 0)
            distribution    = stats.get("communityDistribution", {})

            log(f"  Found {community_count} communities, "
                f"modularity = {modularity:.4f}" if modularity else "")

            # 3. Run Louvain write to compute coverage and conductance
            log("  Writing community IDs to nodes...")
            self._run(
                f"CALL gds.louvain.write('{g}', "
                f"{{writeProperty: '__community_{uid}'}}) "
                f"YIELD communityCount"
            )

            # 4. Compute Coverage:
            #    = nodes in communities of size >= 2 / total nodes
            log("  Computing coverage...")
            cov_result = self._one(f"""
                MATCH (n:{node_label})
                WHERE n.__community_{uid} IS NOT NULL
                WITH n.__community_{uid} AS cid, count(*) AS sz
                WITH sum(CASE WHEN sz >= 2 THEN sz ELSE 0 END) AS in_community,
                     sum(sz) AS total
                RETURN toFloat(in_community) / total AS coverage
            """)
            coverage = cov_result.get("coverage", 0.0)

            # 5. Compute Conductance:
            #    For each community: external_edges / (2 * internal_edges + external_edges)
            #    Then average across communities. Lower = better partitioning.
            log("  Computing conductance...")
            cond_result = self._one(f"""
                MATCH (n:{node_label})-[r:{rel_type}]-(m:{node_label})
                WHERE n.__community_{uid} IS NOT NULL
                  AND m.__community_{uid} IS NOT NULL
                WITH n.__community_{uid} AS cid,
                     count(CASE WHEN n.__community_{uid} = m.__community_{uid}
                                THEN 1 END) AS internal,
                     count(CASE WHEN n.__community_{uid} <> m.__community_{uid}
                                THEN 1 END) AS external
                WHERE internal + external > 0
                WITH cid,
                     toFloat(external) / (internal + external) AS cond
                RETURN avg(cond) AS avg_conductance
            """)
            conductance = cond_result.get("avg_conductance", 0.0)

            # 6. Cleanup the written property
            log("  Cleaning up temporary properties...")
            self._run(f"""
                MATCH (n:{node_label})
                WHERE n.__community_{uid} IS NOT NULL
                REMOVE n.__community_{uid}
            """)

            metrics = {
                "Modularity":  round(float(modularity), 4) if modularity is not None else None,
                "Coverage":    round(float(coverage), 4)    if coverage is not None else None,
                "Conductance": round(float(conductance), 4) if conductance is not None else None,
            }

            elapsed = round(time.time() - ts, 1)
            log(f"  ✓ Done in {elapsed}s")

            return {"success": True, "metrics": metrics, "elapsed": elapsed}

        except Exception as e:
            return {
                "success": False, "metrics": {},
                "error": str(e), "elapsed": round(time.time() - ts, 1),
            }
        finally:
            self._drop(g)

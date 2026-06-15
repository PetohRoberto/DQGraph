"""
link_prediction.py
------------------
Link Prediction via Neo4j Graph Data Science (GDS).

Features supportées :
  - Heuristiques (aucune propriété requise) :
      COMMON_NEIGHBORS, ADAMIC_ADAR, RESOURCE_ALLOCATION,
      PREFERENTIAL_ATTACHMENT, TOTAL_NEIGHBORS
  - Embedding (nécessite un addNodeProperty FastRP préalable) :
      HADAMARD, L2, COSINE  →  activées uniquement si use_embeddings=True

Modèles : logistic_regression (défaut), random_forest
"""

from __future__ import annotations
from neo4j import GraphDatabase
import time

# ── constantes ────────────────────────────────────────────────────────────────

# All link prediction features require nodeProperties (embeddings).
LP_FEATURES = ["L2", "HADAMARD", "COSINE", "SAME_CATEGORY"]
_LP_FEATURE_SET = set(LP_FEATURES)


class LinkPredictionPipeline:

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _run(self, query: str, params: dict = None) -> list[dict]:
        with self.driver.session(database=self.database) as s:
            return s.run(query, **(params or {})).data()

    def _one(self, query: str, params: dict = None) -> dict:
        with self.driver.session(database=self.database) as s:
            r = s.run(query, **(params or {})).single()
            return dict(r) if r else {}

    def _silent_drop(self, name: str, kind: str = "graph"):
        try:
            if kind == "graph":
                self._run(f"CALL gds.graph.drop('{name}', false)")
            else:
                self._run(f"CALL gds.pipeline.drop('{name}', false)")
        except Exception:
            pass

    # ── schema helpers ────────────────────────────────────────────────────────

    def get_node_labels(self) -> list[str]:
        return [r["label"] for r in
                self._run("CALL db.labels() YIELD label RETURN label ORDER BY label")]

    def get_relationship_types(self) -> list[str]:
        return [r["relationshipType"] for r in self._run(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType ORDER BY relationshipType")]

    def get_node_labels_for_rel(self, rel_type: str):
        rows = self._run(f"""
            MATCH (s)-[:{rel_type}]->(d)
            WITH DISTINCT labels(s) AS sl, labels(d) AS dl LIMIT 50
            RETURN sl, dl
        """)
        src, dst = set(), set()
        for r in rows:
            src |= set(r["sl"]); dst |= set(r["dl"])
        return sorted(src), sorted(dst)

    def count_nodes_and_edges(self, src: str, rel: str, dst: str) -> dict:
        return self._one(f"""
            MATCH (s:{src})
            OPTIONAL MATCH (s)-[r:{rel}]->(d:{dst})
            RETURN count(DISTINCT s) AS nodes, count(r) AS edges
        """)

    def check_gds_available(self) -> bool:
        try:
            self._run("RETURN gds.version() AS v")
            return True
        except Exception:
            return False

    # ── GDS steps ─────────────────────────────────────────────────────────────

    def _project(self, graph_name: str, src: str, dst: str,
                 rel: str, node_props: list[str] = None) -> dict:
        self._silent_drop(graph_name, "graph")
        labels = list({src, dst})
        nodes  = ({l: {"properties": node_props} for l in labels}
                  if node_props else labels)
        rels   = {rel: {"orientation": "UNDIRECTED"}}
        return self._one(
            "CALL gds.graph.project($n, $nodes, $rels) "
            "YIELD graphName, nodeCount, relationshipCount",
            {"n": graph_name, "nodes": nodes, "rels": rels},
        )

    def _create_pipeline(self, pipe: str) -> dict:
        self._silent_drop(pipe, "pipeline")
        return self._one(
            "CALL gds.beta.pipeline.linkPrediction.create($p)",
            {"p": pipe},
        )

    def _add_fastrp(self, pipe: str, graph_name: str,
                    embedding_dim: int = 64) -> dict:
        """
        Ajoute un step FastRP pour générer des embeddings sur les nœuds.
        Requis avant d'utiliser HADAMARD / L2 / COSINE.
        """
        return self._one(
            """
            CALL gds.beta.pipeline.linkPrediction.addNodeProperty(
                $pipe,
                'fastRP',
                {
                    mutateProperty:  'embedding',
                    embeddingDimension: $dim,
                    randomSeed: 42
                }
            ) YIELD name
            """,
            {"pipe": pipe, "dim": embedding_dim},
        )

    def _add_features(self, pipe: str, features: list[str]) -> None:
        """
        Add link features. FastRP embeddings MUST be added first via _add_fastrp.
        All features use nodeProperties: ['embedding'].
        Supported: L2, HADAMARD, COSINE, SAME_CATEGORY.
        """
        for feat in features:
            f = feat.upper()
            if f not in _LP_FEATURE_SET:
                raise ValueError(
                    f"Feature '{feat}' unknown. Supported: {LP_FEATURES}."
                )
            self._one(
                "CALL gds.beta.pipeline.linkPrediction.addFeature("
                "$p, $f, {nodeProperties: ['embedding']}) YIELD featureSteps",
                {"p": pipe, "f": f},
            )

    def _configure_split(self, pipe: str, test_frac: float,
                         train_frac: float, neg_ratio: float) -> dict:
        return self._one(
            "CALL gds.beta.pipeline.linkPrediction.configureSplit("
            "$p, {testFraction: $tf, trainFraction: $trf, "
            "negativeSamplingRatio: $nsr}) YIELD splitConfig",
            {"p": pipe, "tf": test_frac, "trf": train_frac, "nsr": neg_ratio},
        )

    def _add_model(self, pipe: str, algorithm: str) -> dict:
        if algorithm == "random_forest":
            return self._one(
                "CALL gds.beta.pipeline.linkPrediction.addRandomForest("
                "$p, {}) YIELD parameterSpace",
                {"p": pipe},
            )
        return self._one(
            "CALL gds.beta.pipeline.linkPrediction.addLogisticRegression("
            "$p) YIELD parameterSpace",
            {"p": pipe},
        )

    def _train(self, graph: str, pipe: str, model: str,
               rel: str, src: str, dst: str,
               metrics: list[str], concurrency: int) -> dict:
        try:
            self._run(f"CALL gds.model.drop('{model}', false)")
        except Exception:
            pass
        return self._one(
            """
            CALL gds.beta.pipeline.linkPrediction.train($graph, {
                pipeline:               $pipe,
                modelName:              $model,
                targetRelationshipType: $rel,
                sourceNodeLabel:        $src,
                targetNodeLabel:        $dst,
                metrics:                $metrics,
                concurrency:            $conc
            })
            YIELD modelInfo, modelSelectionStats
            RETURN modelInfo, modelSelectionStats
            """,
            {"graph": graph, "pipe": pipe, "model": model,
             "rel": rel, "src": src, "dst": dst,
             "metrics": metrics, "conc": concurrency},
        )

    def _predict(self, graph: str, model: str,
                 top_n: int, threshold: float) -> list[dict]:
        rows = self._run(
            """
            CALL gds.beta.pipeline.linkPrediction.predict.stream(
                $graph, {modelName: $model, topN: $n, threshold: $t}
            )
            YIELD node1, node2, probability
            RETURN gds.util.asNode(node1) AS n1,
                   gds.util.asNode(node2) AS n2,
                   probability
            ORDER BY probability DESC
            """,
            {"graph": graph, "model": model, "n": top_n, "t": threshold},
        )
        return [
            {"node1": dict(r["n1"]) if r["n1"] else {},
             "node2": dict(r["n2"]) if r["n2"] else {},
             "probability": round(r["probability"], 4)}
            for r in rows
        ]

    def _cleanup(self, graph: str, pipe: str, model: str):
        self._silent_drop(graph, "graph")
        self._silent_drop(pipe,  "pipeline")
        try:
            self._run(f"CALL gds.model.drop('{model}', false)")
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        src_label:      str,
        dst_label:      str,
        rel_type:       str,
        features:       list[str],
        algorithm:      str   = "logistic_regression",
        test_fraction:  float = 0.2,
        train_fraction: float = 0.6,
        top_n:          int   = 20,
        threshold:      float = 0.3,
        node_props:     list[str] = None,
        embedding_dim:  int   = 64,
        progress_callback = None,
    ) -> dict:
        """
        Lance le pipeline complet.

        Si les features contiennent HADAMARD / L2 / COSINE, un step FastRP est
        ajouté automatiquement pour générer les embeddings — pas besoin de
        pré-calculer quoi que ce soit.

        Retourne un dict :
          success, graph_stats, model_info, model_selection,
          predictions, elapsed, features_used, algorithm, params
        """
        ts  = time.time()
        uid = str(int(ts))[-6:]
        g   = f"lp_graph_{uid}"
        p   = f"lp_pipe_{uid}"
        m   = f"lp_model_{uid}"

        def log(msg):
            if progress_callback:
                progress_callback(msg)

        features_upper = [f.upper() for f in features]

        # Validate features
        for f in features_upper:
            if f not in _LP_FEATURE_SET:
                return {'success': False, 'error': f'Unknown feature: {f}', 'elapsed': 0}

        try:
            log("📐 Projection du graphe en mémoire...")
            graph_stats = self._project(g, src_label, dst_label, rel_type, node_props)

            log("🔧 Création du pipeline GDS...")
            self._create_pipeline(p)

            log("🔢 Génération des embeddings FastRP...")
            self._add_fastrp(p, g, embedding_dim)

            log("➕ Ajout des features...")
            self._add_features(p, features_upper)

            log("⚙️ Configuration du split train/test...")
            self._configure_split(p, test_fraction, train_fraction, 1.0)

            log(f"🤖 Ajout du modèle ({algorithm})...")
            self._add_model(p, algorithm)

            log("🚀 Entraînement en cours...")
            train_result = self._train(g, p, m, rel_type,
                                       src_label, dst_label,
                                       ["AUCPR"], 4)

            log("🔮 Génération des prédictions...")
            predictions = self._predict(g, m, top_n, threshold)

            elapsed = round(time.time() - ts, 1)
            log(f"✅ Terminé en {elapsed}s — {len(predictions)} prédictions")

            return {
                "success":         True,
                "graph_stats":     graph_stats,
                "model_info":      train_result.get("modelInfo", {}),
                "model_selection": train_result.get("modelSelectionStats", {}),
                "predictions":     predictions,
                "elapsed":         elapsed,
                "features_used":   features_upper,
                "algorithm":       algorithm,
                "params": {
                    "src_label":  src_label,
                    "dst_label":  dst_label,
                    "rel_type":   rel_type,
                    "top_n":      top_n,
                    "threshold":  threshold,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error":   str(e),
                "elapsed": round(time.time() - ts, 1),
            }
        finally:
            self._cleanup(g, p, m)

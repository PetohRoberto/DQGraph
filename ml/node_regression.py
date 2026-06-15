"""
node_regression.py
------------------
Node Property Regression via Neo4j Graph Data Science (GDS).

Uses gds.alpha.pipeline.nodeRegression (GDS 2.3+).
Falls back to a manual Cypher-based approach if the alpha API is unavailable.

Pipeline :
  1. Project subgraph
  2. Create nodeRegression pipeline
  3. Generate FastRP embeddings
  4. Add LinearRegression
  5. Train & evaluate
  6. Cleanup
"""

from __future__ import annotations
from neo4j import GraphDatabase
import time
import math


class NodeRegressionPipeline:

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

    def _drop(self, name: str, kind: str = "graph"):
        try:
            if kind == "graph":
                self._run(f"CALL gds.graph.drop('{name}', false)")
            else:
                self._run(f"CALL gds.pipeline.drop('{name}', false)")
        except Exception:
            pass

    def _drop_model(self, name: str):
        try:
            self._run(f"CALL gds.model.drop('{name}', false)")
        except Exception:
            pass

    def _check_alpha_available(self) -> bool:
        """Check if gds.alpha.pipeline.nodeRegression is available."""
        try:
            self._run("CALL gds.list() YIELD name "
                      "WHERE name = 'gds.alpha.pipeline.nodeRegression.create' "
                      "RETURN name")
            return True
        except Exception:
            return False

    # ── GDS pipeline approach ─────────────────────────────────────────────────

    def _clean_nan(self, node_label, target_property):
        """Remove NaN/invalid values from target property before training.
        Uses REMOVE (not SET null) and toString() for reliable NaN detection."""
        total = 0
        with self.driver.session(database=self.database) as s:
            r = s.run(f"""
                MATCH (n:{node_label})
                WHERE n.{target_property} IS NOT NULL
                  AND toString(n.{target_property}) = 'NaN'
                REMOVE n.{target_property}
                RETURN count(n) AS c
            """).single()
            total += r["c"] if r else 0
            r = s.run(f"""
                MATCH (n:{node_label})
                WHERE n.{target_property} IS NOT NULL
                  AND toString(n.{target_property}) = ''
                REMOVE n.{target_property}
                RETURN count(n) AS c
            """).single()
            total += r["c"] if r else 0
        return total

    def _run_gds_pipeline(
        self, node_label, rel_type, target_property,
        algorithm, embedding_dim, test_fraction, log
    ) -> dict:
        ts  = time.time()
        uid = str(int(ts))[-6:]
        g = f"nr_graph_{uid}"
        p = f"nr_pipe_{uid}"
        m = f"nr_model_{uid}"

        try:
            log("  Cleaning NaN values from target property...")
            cleaned = self._clean_nan(node_label, target_property)
            if cleaned > 0:
                log(f"  Removed {cleaned} NaN/invalid values")

            log("  Projecting graph...")
            self._drop(g, "graph")
            self._one(
                f"CALL gds.graph.project($g, "
                f"{{{node_label}: {{properties: ['{target_property}']}}}}, "
                f"{{{rel_type}: {{orientation: 'UNDIRECTED'}}}}) "
                f"YIELD graphName, nodeCount, relationshipCount",
                {"g": g},
            )

            log("  Creating regression pipeline...")
            self._drop(p, "pipeline")
            self._one(
                "CALL gds.alpha.pipeline.nodeRegression.create($p)",
                {"p": p},
            )

            log("  Generating FastRP embeddings...")
            self._one(
                f"CALL gds.alpha.pipeline.nodeRegression.addNodeProperty("
                f"$p, 'fastRP', {{mutateProperty: 'embedding', "
                f"embeddingDimension: {embedding_dim}, randomSeed: 42}}) YIELD name",
                {"p": p},
            )

            log("  Adding features...")
            self._one(
                "CALL gds.alpha.pipeline.nodeRegression.selectFeatures("
                "$p, ['embedding']) YIELD name",
                {"p": p},
            )

            log("  Configuring split...")
            self._one(
                "CALL gds.alpha.pipeline.nodeRegression.configureSplit("
                "$p, {testFraction: $tf}) YIELD splitConfig",
                {"p": p, "tf": test_fraction},
            )

            log(f"  Adding model ({algorithm})...")
            self._one(
                "CALL gds.alpha.pipeline.nodeRegression.addLinearRegression("
                "$p, {}) YIELD parameterSpace",
                {"p": p},
            )

            log("  Training...")
            self._drop_model(m)
            result = self._one(
                f"""
                CALL gds.alpha.pipeline.nodeRegression.train($g, {{
                    pipeline:        $p,
                    modelName:       $m,
                    targetNodeLabels: ['{node_label}'],
                    targetProperty:  '{target_property}',
                    metrics:         ['MEAN_SQUARED_ERROR', 'MEAN_ABSOLUTE_ERROR'],
                    concurrency:     4
                }})
                YIELD modelInfo
                RETURN modelInfo
                """,
                {"g": g, "p": p, "m": m},
            )

            # Extract metrics — search all possible GDS structures
            model_info = result.get("modelInfo", {})
            if not isinstance(model_info, dict):
                model_info = {}

            def _dig(obj, key):
                """Recursively search for key in nested dicts."""
                if not isinstance(obj, dict):
                    return None
                if key in obj:
                    v = obj[key]
                    if isinstance(v, dict):
                        for sub in ["test", "avg", "outerTrain"]:
                            if sub in v:
                                try: return float(v[sub])
                                except: pass
                        for sv in v.values():
                            try: return float(sv)
                            except: continue
                        return None
                    try: return float(v)
                    except: return None
                for v in obj.values():
                    found = _dig(v, key)
                    if found is not None:
                        return found
                return None

            mse = _dig(model_info, "MEAN_SQUARED_ERROR")
            mae = _dig(model_info, "MEAN_ABSOLUTE_ERROR")

            # Also check modelSelectionStats
            if mse is None:
                mse = _dig(result, "MEAN_SQUARED_ERROR")
            if mae is None:
                mae = _dig(result, "MEAN_ABSOLUTE_ERROR")

            # Log what we found for debugging
            log(f"  GDS metrics: MSE={mse}, MAE={mae}")
            if mse is None and mae is None:
                log(f"  modelInfo keys: {list(model_info.keys()) if isinstance(model_info, dict) else 'N/A'}")

            metrics = {}
            if mse is not None:
                metrics["RMSE"] = math.sqrt(mse)
            if mae is not None:
                metrics["MAE"] = mae

            # Compute R² from MSE + total variance via Cypher
            if mse is not None:
                try:
                    var_r = self._one(f"""
                        MATCH (n:{node_label})
                        WHERE n.{target_property} IS NOT NULL
                          AND toString(n.{target_property}) <> 'NaN'
                        WITH toFloat(n.{target_property}) AS v
                        WITH avg(v) AS mu, collect(v) AS vals
                        UNWIND vals AS val
                        RETURN avg((val - mu)*(val - mu)) AS variance
                    """)
                    variance = float(var_r.get("variance", 0))
                    if variance > 1e-10:
                        metrics["R²"] = max(0.0, 1.0 - mse / variance)
                    else:
                        metrics["R²"] = 0.0
                except:
                    metrics["R²"] = None
            
            # If GDS trained but returned no extractable metrics, try raw result
            if not metrics:
                log("  ⚠ No metrics extracted from GDS. Trying Cypher evaluation...")
                # Use GDS model to get predictions, then compute metrics manually
                try:
                    pred_r = self._run(f"""
                        CALL gds.beta.pipeline.nodeRegression.predict.stream('{g}', {{
                            modelName: '{m}',
                            targetNodeLabels: ['{node_label}']
                        }})
                        YIELD nodeId, predictedValue
                        WITH gds.util.asNode(nodeId) AS n, predictedValue
                        WHERE n.{target_property} IS NOT NULL
                          AND toString(n.{target_property}) <> 'NaN'
                        WITH toFloat(n.{target_property}) AS actual, predictedValue AS predicted
                        RETURN avg((actual - predicted) * (actual - predicted)) AS mse,
                               avg(abs(actual - predicted)) AS mae,
                               count(*) AS cnt
                    """)
                    if pred_r and pred_r[0].get("cnt", 0) > 0:
                        row = pred_r[0]
                        mse_val = float(row["mse"])
                        mae_val = float(row["mae"])
                        metrics["RMSE"] = math.sqrt(mse_val)
                        metrics["MAE"]  = mae_val
                        # R² from prediction MSE
                        var_r2 = self._one(f"""
                            MATCH (n:{node_label})
                            WHERE n.{target_property} IS NOT NULL
                              AND toString(n.{target_property}) <> 'NaN'
                            WITH toFloat(n.{target_property}) AS v
                            WITH avg(v) AS mu, collect(v) AS vals
                            UNWIND vals AS val
                            RETURN avg((val-mu)*(val-mu)) AS variance
                        """)
                        var_val = float(var_r2.get("variance", 0))
                        metrics["R²"] = max(0.0, 1.0 - mse_val / var_val) if var_val > 1e-10 else 0.0
                        log(f"  ✓ Computed from predictions: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R²']:.4f}")
                except Exception as pred_err:
                    log(f"  ⚠ Prediction fallback failed: {pred_err}")

            elapsed = round(time.time() - ts, 1)
            log(f"  ✓ Done in {elapsed}s")
            return {"success": True, "metrics": metrics, "elapsed": elapsed}

        except Exception as e:
            return {
                "success": False, "metrics": {},
                "error": str(e), "elapsed": round(time.time() - ts, 1),
            }
        finally:
            self._drop(g, "graph")
            self._drop(p, "pipeline")
            self._drop_model(m)

    # ── Cypher fallback approach ──────────────────────────────────────────────

    def _run_cypher_fallback(
        self, node_label, target_property, log
    ) -> dict:
        """
        Fallback: compute regression metrics using Cypher only.
        Uses the graph mean as a naive predictor, then computes MAE, RMSE, R².
        This gives a baseline that degrades with data quality.
        """
        ts = time.time()
        log("  Using Cypher-based regression (mean predictor)...")

        try:
            with self.driver.session(database=self.database) as session:
                # Get all values and the mean
                result = session.run(f"""
                    MATCH (n:{node_label})
                    WHERE n.{target_property} IS NOT NULL
                      AND toFloat(n.{target_property}) IS NOT NULL
                    WITH toFloat(n.{target_property}) AS val
                    WITH collect(val) AS vals, avg(val) AS mean_val
                    UNWIND vals AS v
                    WITH v, mean_val,
                         (v - mean_val) * (v - mean_val) AS sq_error,
                         abs(v - mean_val) AS abs_error
                    RETURN avg(sq_error) AS mse,
                           avg(abs_error) AS mae,
                           count(v) AS n,
                           mean_val
                """).single()

                if result and result["n"] > 0:
                    mse = float(result["mse"])
                    mae = float(result["mae"])

                    # Also compute total variance for R²
                    var_result = session.run(f"""
                        MATCH (n:{node_label})
                        WHERE n.{target_property} IS NOT NULL
                        WITH toFloat(n.{target_property}) AS val
                        WITH avg(val) AS mu, collect(val) AS vals
                        UNWIND vals AS v
                        RETURN avg((v - mu)*(v - mu)) AS variance
                    """).single()

                    variance = float(var_result["variance"]) if var_result else mse
                    r_squared = max(0.0, 1.0 - mse / max(variance, 1e-10))

                    metrics = {
                        "RMSE": math.sqrt(mse),
                        "MAE":  mae,
                        "R²":   r_squared,
                    }
                else:
                    metrics = {"RMSE": None, "MAE": None, "R²": None}

            elapsed = round(time.time() - ts, 1)
            log(f"  ✓ Fallback done in {elapsed}s")
            return {"success": True, "metrics": metrics, "elapsed": elapsed}

        except Exception as e:
            return {
                "success": False, "metrics": {},
                "error": str(e), "elapsed": round(time.time() - ts, 1),
            }

    # ── Public API ────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        node_label: str,
        rel_type: str,
        target_property: str,
        algorithm: str = "linear_regression",
        embedding_dim: int = 64,
        test_fraction: float = 0.2,
        progress_callback=None,
    ) -> dict:
        """
        Run node regression. Tries GDS alpha pipeline first, falls back to
        Cypher-based computation if unavailable.

        Returns dict with: success, metrics (RMSE, MAE, R²), elapsed
        """
        def log(msg):
            if progress_callback:
                progress_callback(msg)

        if self._check_alpha_available():
            return self._run_gds_pipeline(
                node_label, rel_type, target_property,
                algorithm, embedding_dim, test_fraction, log
            )
        else:
            return self._run_cypher_fallback(
                node_label, target_property, log
            )

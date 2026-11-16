import random
import statistics as _statistics
from neo4j_pool import get_driver
from value_types import classify
from typing import Literal

ElementType = Literal["node", "edge"]

SEED = 42

# Appended to a value to break its conformity: "1995-11-22" -> "1995-11-22@1".
NONCONFORM_SUFFIX = "@1"

BATCH = 5000


def _label_expr(labels) -> str:
    """':`A`:`B`' for a CREATE pattern (backticks in a label are doubled)."""
    return "".join(":`" + l.replace("`", "``") + "`" for l in labels)


def _element_match(element_type, label_or_type, src_label=None, dst_label=None) -> str:
    """MATCH clause binding the degraded elements to `x`."""
    if element_type == "node":
        return f"MATCH (x:`{label_or_type}`)"
    src_f = f":`{src_label}`" if src_label else ""
    dst_f = f":`{dst_label}`" if dst_label else ""
    return f"MATCH ({src_f})-[x:`{label_or_type}`]->({dst_f})"


def _property_samples(session, match: str, n: int = 10) -> list[dict]:
    """[{prop, samples}] for EVERY property key of the elements `x` matched by
    `match`, with up to n non-null values each. The keys come from a full scan,
    not from the first instances: a property carried by a few instances only
    is degraded too ("for every property of the selected label")."""
    keys = [r["k"] for r in session.run(
        f"{match} UNWIND keys(x) AS k RETURN DISTINCT k")]
    if n == 0:   # keys only
        return [{"prop": k, "samples": []} for k in keys]
    return [{"prop": k, "samples": [r["v"] for r in session.run(
                f"{match} WHERE x[$k] IS NOT NULL RETURN x[$k] AS v LIMIT $n",
                k=k, n=n)]}
            for k in keys]


class DataCompletenessDegrader:
    """
    Méthodes de dégradation de la qualité (complétude, couverture des arcs,
    conformité, doublons, exactitude).
    Toutes les opérations sont réversibles si l'on conserve les valeurs originales
    (les doublons créés sont supprimés par remove_duplicates).

    Les tirages utilisent un générateur initialisé avec `seed` sur des instances
    triées par elementId : à graphe identique, la dégradation est reproductible.
    Pour une expérience à plusieurs niveaux λ, utiliser un seul degrader : il
    mémorise les instances déjà altérées (accuracy, conformity, duplicates).
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 seed: int = SEED):
        self.driver = get_driver(uri, user, password)
        self.database = database
        self.rng = random.Random(seed)
        # (element_type, label, src, dst, key) -> σ of the original values /
        # elementIds already noised, so accuracy degradation stays cumulative.
        self._sigma: dict[tuple, float] = {}
        self._noisy: dict[tuple, set] = {}
        # Same key -> elementIds already made non-conforming.
        self._nonconform: dict[tuple, set] = {}
        # (src, rel, dst) -> original edge elementIds / those already duplicated;
        # elementIds of the node copies, deleted by remove_duplicates().
        self._dup_pool: dict[tuple, list] = {}
        self._dup_done: dict[tuple, set] = {}
        self._dup_nodes: list[str] = []
        # Where each copy comes from, so that the ML split can put a copy on
        # the same side (train / test) as its original:
        # copy node id -> original node id, (copy src, copy dst) -> (src, dst).
        self.dup_origin_nodes: dict[str, str] = {}
        self.dup_origin_edges: dict[tuple, tuple] = {}

    def close(self):
        pass  # shared driver, see neo4j_pool

    # ─────────────────────────────────────────────────────────────────────────
    # PropertyCompleteness degradation
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_property_completeness(
        self,
        element_type: ElementType,
        label_or_type: str,
        property_name: str,
        lambda_pc: float,
        src_label: str = None,
        dst_label: str = None,
    ) -> dict:
        """
        Dégrade PropertyCompleteness(p, t) jusqu'à atteindre un taux de valeurs
        manquantes de lambda_pc.

        Pour chaque instance ayant la propriété p, on remplace la valeur par :
          - NaN   (float)  pour les valeurs numériques
          - ""    (string) pour les chaînes de caractères

        Si le taux actuel de valeurs manquantes est déjà >= lambda_pc, aucune
        modification n'est effectuée.

        Paramètres
        ----------
        element_type   : "node" ou "edge"
        label_or_type  : label du nœud ou type de la relation
        property_name  : propriété à dégrader
        lambda_pc      : proportion cible de valeurs manquantes ∈ [0, 1]
        src_label      : (arcs seulement) label du nœud source
        dst_label      : (arcs seulement) label du nœud destination

        Retourne
        --------
        dict avec les clés :
          - total          : nombre total d'instances
          - already_missing: instances déjà sans la propriété avant dégradation
          - degraded       : nombre d'instances effectivement dégradées
          - final_missing  : nombre total de valeurs manquantes après dégradation
        """
        if not 0.0 <= lambda_pc <= 1.0:
            raise ValueError("lambda_pc doit être compris entre 0 et 1.")

        with self.driver.session(database=self.database) as session:

            # 1. Récupérer le total d'instances et celles qui ont déjà la propriété
            if element_type == "node":
                count_query = f"""
                MATCH (n:{label_or_type})
                RETURN count(n) AS total,
                       count(n.{property_name}) AS with_prop
                """
            else:
                src_f = f":{src_label}" if src_label else ""
                dst_f = f":{dst_label}" if dst_label else ""
                count_query = f"""
                MATCH ({src_f})-[e:{label_or_type}]->({dst_f})
                RETURN count(e) AS total,
                       count(e.{property_name}) AS with_prop
                """
            counts = session.run(count_query).single()
            total      = counts["total"]
            with_prop  = counts["with_prop"]
            already_missing = total - with_prop

            if total == 0:
                return {"total": 0, "already_missing": 0, "degraded": 0, "final_missing": 0}

            # 2. Calculer combien d'instances supplémentaires à dégrader
            target_missing = int(lambda_pc * total)
            to_degrade = max(0, target_missing - already_missing)

            if to_degrade == 0:
                return {
                    "total": total,
                    "already_missing": already_missing,
                    "degraded": 0,
                    "final_missing": already_missing,
                }

            # 3. Récupérer les ids des instances qui ont encore la propriété
            if element_type == "node":
                ids_query = f"""
                MATCH (n:{label_or_type})
                WHERE n.{property_name} IS NOT NULL
                RETURN elementId(n) AS eid, n.{property_name} AS val
                """
            else:
                ids_query = f"""
                MATCH ({src_f})-[e:{label_or_type}]->({dst_f})
                WHERE e.{property_name} IS NOT NULL
                RETURN elementId(e) AS eid, e.{property_name} AS val
                """
            rows = sorted(session.run(ids_query).data(), key=lambda r: r["eid"])

            # 4. Tirage aléatoire
            sample = self.rng.sample(rows, min(to_degrade, len(rows)))

            # 5. Apply degradation: REMOVE the property (null = missing)
            #    This avoids NaN issues with GDS and Neo4j.
            #    One batched query instead of one round-trip per instance.
            if element_type == "node":
                upd_query = f"""
                UNWIND $eids AS eid
                MATCH (n) WHERE elementId(n) = eid
                REMOVE n.{property_name}
                """
            else:
                upd_query = f"""
                UNWIND $eids AS eid
                MATCH ()-[e]->() WHERE elementId(e) = eid
                REMOVE e.{property_name}
                """
            session.run(upd_query, eids=[row["eid"] for row in sample])
            degraded_count = len(sample)

            return {
                "total": total,
                "already_missing": already_missing,
                "degraded": degraded_count,
                "final_missing": already_missing + degraded_count,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # EdgeCoverage degradation
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_edge_coverage(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        lambda_ec: float,
        scope_ids: set = None,
    ) -> dict:
        """
        Dégrade EdgeCoverage(t1, t2) jusqu'à atteindre un taux de nœuds sans arc
        de lambda_ec.
        scope_ids : nœuds source pouvant être dégradés (p. ex. ceux de
        l'ensemble d'entraînement) ; lambda_ec est alors une proportion de ces
        nœuds. None = tous les nœuds source.

        Pour chaque nœud source ayant au moins un arc rel_type vers un nœud
        dst_label, on supprime tous ses arcs de ce type (sans supprimer les
        nœuds source ni destination) jusqu'à atteindre la proportion cible.

        Paramètres
        ----------
        src_label  : label du nœud source  (t1)
        rel_type   : type de la relation   (t2)
        dst_label  : label du nœud destination (t3)
        lambda_ec  : proportion cible de nœuds source SANS arc ∈ [0, 1]

        Retourne
        --------
        dict avec les clés :
          - total_src       : nombre total de nœuds source
          - already_no_edge : nœuds source déjà sans arc avant dégradation
          - degraded_nodes  : nœuds source dont les arcs ont été supprimés
          - edges_deleted   : nombre total d'arcs supprimés
          - final_no_edge   : nombre de nœuds source sans arc après dégradation
        """
        if not 0.0 <= lambda_ec <= 1.0:
            raise ValueError("lambda_ec doit être compris entre 0 et 1.")

        with self.driver.session(database=self.database) as session:

            # 1. Compter le total de nœuds source et ceux qui ont déjà l'arc
            counts = session.run(f"""
                MATCH (src:{src_label})
                OPTIONAL MATCH (src)-[r:{rel_type}]->(dst:{dst_label})
                WITH src, count(r) AS nb_edges
                RETURN count(src) AS total_src,
                       count(CASE WHEN nb_edges = 0 THEN 1 END) AS already_no_edge
            """).single()

            total_src       = counts["total_src"]
            already_no_edge = counts["already_no_edge"]

            # Source nodes that still have at least one edge
            rows = sorted(session.run(f"""
                MATCH (src:{src_label})-[:{rel_type}]->(:{dst_label})
                RETURN DISTINCT elementId(src) AS eid
            """).data(), key=lambda r: r["eid"])
            if scope_ids is not None:   # counts restricted to the scope
                rows = [r for r in rows if r["eid"] in scope_ids]
                total_src = len(scope_ids)
                already_no_edge = total_src - len(rows)

            if total_src == 0:
                return {"total_src": 0, "already_no_edge": 0,
                        "degraded_nodes": 0, "edges_deleted": 0, "final_no_edge": 0}

            # 2. Calculer combien de nœuds supplémentaires à priver d'arcs
            target_no_edge = int(lambda_ec * total_src)
            to_degrade = max(0, target_no_edge - already_no_edge)

            if to_degrade == 0:
                return {
                    "total_src": total_src,
                    "already_no_edge": already_no_edge,
                    "degraded_nodes": 0,
                    "edges_deleted": 0,
                    "final_no_edge": already_no_edge,
                }

            # 3. Tirer parmi les nœuds source qui ont encore au moins un arc
            sample = self.rng.sample(rows, min(to_degrade, len(rows)))
            sampled_ids = [r["eid"] for r in sample]

            # 4. Supprimer les arcs (sans toucher aux nœuds)
            result = session.run(f"""
                MATCH (src:{src_label})-[r:{rel_type}]->(dst:{dst_label})
                WHERE elementId(src) IN $ids
                DELETE r
                RETURN count(r) AS deleted
            """, ids=sampled_ids).single()

            edges_deleted = result["deleted"] if result else 0

            return {
                "total_src": total_src,
                "already_no_edge": already_no_edge,
                "degraded_nodes": len(sampled_ids),
                "edges_deleted": edges_deleted,
                "final_no_edge": already_no_edge + len(sampled_ids),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Property Accuracy Degradation (Gaussian noise)
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_property_accuracy(
        self,
        element_type: ElementType,
        label_or_type: str,
        property_name: str,
        lambda_num: float,
        src_label: str = None,
        dst_label: str = None,
    ) -> dict:
        """
        Degrades PropertyAccuracy by adding Gaussian noise to numeric values.

        For each instance x:  π̃(x, p) = π(x, p) + ε,  ε ~ N(0, λ_num²)

        Parameters
        ----------
        element_type   : "node" or "edge"
        label_or_type  : label or relationship type
        property_name  : numeric property to corrupt
        lambda_num     : standard deviation of the noise (0 = no change)
        src_label      : (edges only) source label
        dst_label      : (edges only) destination label

        Returns
        -------
        dict with: total, corrupted, lambda_num
        """
        if lambda_num <= 0:
            return {"total": 0, "corrupted": 0, "lambda_num": 0}

        with self.driver.session(database=self.database) as session:

            # 1. Fetch all numeric values
            if element_type == "node":
                query = f"""
                MATCH (n:{label_or_type})
                WHERE n.{property_name} IS NOT NULL
                  AND toString(n.{property_name}) <> 'NaN'
                RETURN elementId(n) AS eid, toFloat(n.{property_name}) AS val
                """
            else:
                src_f = f":{src_label}" if src_label else ""
                dst_f = f":{dst_label}" if dst_label else ""
                query = f"""
                MATCH ({src_f})-[e:{label_or_type}]->({dst_f})
                WHERE e.{property_name} IS NOT NULL
                  AND toString(e.{property_name}) <> 'NaN'
                RETURN elementId(e) AS eid, toFloat(e.{property_name}) AS val
                """
            rows = sorted(session.run(query).data(), key=lambda r: r["eid"])

            if not rows:
                return {"total": 0, "corrupted": 0, "lambda_num": lambda_num}

            # 2. Add Gaussian noise to each value
            updates = [
                {"eid": row["eid"], "val": row["val"] + self.rng.gauss(0, lambda_num)}
                for row in rows if row["val"] is not None
            ]
            if element_type == "node":
                upd = f"""
                UNWIND $rows AS row
                MATCH (n) WHERE elementId(n) = row.eid
                SET n.{property_name} = row.val
                """
            else:
                upd = f"""
                UNWIND $rows AS row
                MATCH ()-[e]->() WHERE elementId(e) = row.eid
                SET e.{property_name} = row.val
                """
            session.run(upd, rows=updates)
            corrupted = len(updates)

            return {
                "total": len(rows),
                "corrupted": corrupted,
                "lambda_num": lambda_num,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # All-properties completeness degradation
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_all_completeness(
        self,
        element_type: ElementType,
        label_or_type: str,
        lambda_pc: float,
        src_label: str = None,
        dst_label: str = None,
        exclude_props: list = None,
        scope_ids: set = None,
    ) -> dict:
        """
        For EVERY property key of the element type (except exclude_props, e.g.
        the ML target), removes values so that lambda_pc% of instances have
        that property missing (cumulative target).
        Randomly selects which instances to affect for each key.
        scope_ids: elementIds that may be degraded (e.g. the training nodes);
        lambda_pc is then a proportion of them. None = every instance.
        """
        if not 0.0 <= lambda_pc <= 1.0:
            raise ValueError("lambda_pc doit être compris entre 0 et 1.")

        src_f = f":`{src_label}`" if src_label else ""
        dst_f = f":`{dst_label}`" if dst_label else ""

        with self.driver.session(database=self.database) as session:
            rows = _property_samples(session, _element_match(
                element_type, label_or_type, src_label, dst_label), n=0)

            excluded = set(exclude_props or [])
            all_keys = sorted(r["prop"] for r in rows if r["prop"] not in excluded)
            if not all_keys or lambda_pc == 0:
                return {"total_properties": len(all_keys), "degraded_properties": all_keys,
                        "total_removed": 0}

            total_removed = 0
            for key in all_keys:
                # How many instances currently lack this property?
                if element_type == "node":
                    counts = session.run(f"""
                        MATCH (n:`{label_or_type}`)
                        RETURN count(n) AS total, count(n[$key]) AS with_prop
                    """, key=key).single()
                else:
                    counts = session.run(f"""
                        MATCH ({src_f})-[e:`{label_or_type}`]->({dst_f})
                        RETURN count(e) AS total, count(e[$key]) AS with_prop
                    """, key=key).single()

                total       = counts["total"]
                with_prop   = counts["with_prop"]
                node_rows   = None
                if scope_ids is not None:   # counts restricted to the scope
                    node_rows = [r for r in self._ids_with_key(
                        session, element_type, label_or_type, src_f, dst_f, key)
                        if r["eid"] in scope_ids]
                    total, with_prop = len(scope_ids), len(node_rows)
                already_missing = total - with_prop
                to_remove   = max(0, int(lambda_pc * total) - already_missing)

                if to_remove == 0:
                    continue

                if node_rows is None:
                    node_rows = self._ids_with_key(
                        session, element_type, label_or_type, src_f, dst_f, key)

                node_rows.sort(key=lambda r: r["eid"])
                sample_ids = [r["eid"] for r in
                              self.rng.sample(node_rows, min(to_remove, len(node_rows)))]

                if element_type == "node":
                    session.run(f"""
                        MATCH (n:`{label_or_type}`)
                        WHERE elementId(n) IN $ids
                        SET n[$key] = null
                    """, ids=sample_ids, key=key)
                else:
                    session.run(f"""
                        MATCH ({src_f})-[e:`{label_or_type}`]->({dst_f})
                        WHERE elementId(e) IN $ids
                        SET e[$key] = null
                    """, ids=sample_ids, key=key)

                total_removed += len(sample_ids)

            return {
                "total_properties": len(all_keys),
                "degraded_properties": all_keys,
                "total_removed": total_removed,
            }

    @staticmethod
    def _ids_with_key(session, element_type, label_or_type, src_f, dst_f, key):
        if element_type == "node":
            return session.run(f"""
                MATCH (n:`{label_or_type}`)
                WHERE n[$key] IS NOT NULL
                RETURN elementId(n) AS eid
            """, key=key).data()
        return session.run(f"""
            MATCH ({src_f})-[e:`{label_or_type}`]->({dst_f})
            WHERE e[$key] IS NOT NULL
            RETURN elementId(e) AS eid
        """, key=key).data()

    # ─────────────────────────────────────────────────────────────────────────
    # All-numeric-properties accuracy degradation
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_all_accuracy(
        self,
        element_type: ElementType,
        label_or_type: str,
        lambda_acc: float,
        src_label: str = None,
        dst_label: str = None,
        exclude_props: list = None,
        scope_ids: set = None,
    ) -> dict:
        """
        For EVERY numeric property key (except exclude_props, e.g. the ML
        target), adds Gaussian noise ε ~ N(0, σ_p²) so that a proportion
        lambda_acc of the instances carry a noisy value (cumulative target).

        σ_p is the std of the ORIGINAL values (computed on the first call) and
        the noisy instances are remembered: moving from λ to λ' only noises
        new instances, each instance is noised at most once.
        scope_ids: elementIds that may be degraded; lambda_acc is then a
        proportion of them (σ_p still comes from all the values). None = all.
        """
        if not 0.0 <= lambda_acc <= 1.0:
            raise ValueError("lambda_acc doit être compris entre 0 et 1.")
        if lambda_acc == 0:
            return {"total_numeric_properties": 0, "degraded_properties": [], "total_corrupted": 0}

        src_f = f":`{src_label}`" if src_label else ""
        dst_f = f":`{dst_label}`" if dst_label else ""

        with self.driver.session(database=self.database) as session:
            # Identify numeric property keys by sampling values
            type_rows = _property_samples(session, _element_match(
                element_type, label_or_type, src_label, dst_label))

            # Numbers stored as strings count too (value_types); dates don't.
            excluded = set(exclude_props or [])
            numeric_keys = sorted(
                r["prop"] for r in type_rows
                if r["prop"] not in excluded
                and classify(r["samples"], len(set(map(str, r["samples"]))))
                    in ("numeric", "numeric_discrete")
            )

            if not numeric_keys:
                return {"total_numeric_properties": 0, "degraded_properties": [], "total_corrupted": 0}

            total_corrupted = 0
            for key in numeric_keys:
                if element_type == "node":
                    val_rows = session.run(f"""
                        MATCH (n:`{label_or_type}`)
                        WHERE n[$key] IS NOT NULL
                        RETURN elementId(n) AS eid, toFloat(n[$key]) AS val,
                               n[$key] AS raw
                    """, key=key).data()
                else:
                    val_rows = session.run(f"""
                        MATCH ({src_f})-[e:`{label_or_type}`]->({dst_f})
                        WHERE e[$key] IS NOT NULL
                        RETURN elementId(e) AS eid, toFloat(e[$key]) AS val,
                               e[$key] AS raw
                    """, key=key).data()

                valid_rows = sorted((r for r in val_rows if r["val"] is not None),
                                    key=lambda r: r["eid"])
                if not valid_rows:
                    continue

                state_key = (element_type, label_or_type, src_label, dst_label, key)
                if state_key not in self._sigma:   # first call: values are original
                    all_vals = [r["val"] for r in valid_rows]
                    self._sigma[state_key] = (_statistics.stdev(all_vals) if len(all_vals) > 1
                                              else abs(all_vals[0]) * 0.1 or 1.0)
                noise_std = self._sigma[state_key]
                noisy_ids = self._noisy.setdefault(state_key, set())
                if scope_ids is not None:
                    valid_rows = [r for r in valid_rows if r["eid"] in scope_ids]

                # Only the instances still missing to reach lambda_acc are noised.
                to_corrupt = int(lambda_acc * len(valid_rows)) - len(noisy_ids)
                if to_corrupt <= 0:
                    continue
                clean_rows = [r for r in valid_rows if r["eid"] not in noisy_ids]
                sample = self.rng.sample(clean_rows, min(to_corrupt, len(clean_rows)))
                noisy_ids.update(r["eid"] for r in sample)

                # Keep the storage type: a number stored as a string stays a string.
                updates = []
                for row in sample:
                    noisy = row["val"] + self.rng.gauss(0, noise_std)
                    updates.append({"eid": row["eid"],
                                    "val": str(noisy) if isinstance(row["raw"], str) else noisy})
                if element_type == "node":
                    session.run("""
                        UNWIND $rows AS row
                        MATCH (n) WHERE elementId(n) = row.eid
                        SET n[$key] = row.val
                    """, rows=updates, key=key)
                else:
                    session.run("""
                        UNWIND $rows AS row
                        MATCH ()-[e]->() WHERE elementId(e) = row.eid
                        SET e[$key] = row.val
                    """, rows=updates, key=key)
                total_corrupted += len(updates)

            return {
                "total_numeric_properties": len(numeric_keys),
                "degraded_properties": numeric_keys,
                "total_corrupted": total_corrupted,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # All-properties conformity degradation
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_all_conformity(
        self,
        element_type: ElementType,
        label_or_type: str,
        lambda_conf: float,
        src_label: str = None,
        dst_label: str = None,
        exclude_props: list = None,
        scope_ids: set = None,
    ) -> dict:
        """
        For EVERY scalar property key (except exclude_props, e.g. the ML target),
        a proportion lambda_conf of the instances that carry it have their value
        v replaced by the string v + NONCONFORM_SUFFIX ("1995-11-22" becomes
        "1995-11-22@1"), which no longer matches the expected format.

        Cumulative target: moving from λ to λ' only alters new instances, each
        instance is altered at most once. For a property that fully conformed
        before, lambda_conf = 1 - CondPropertyConformity. List values are skipped.
        scope_ids: elementIds that may be degraded; lambda_conf is then a
        proportion of them. None = every instance.
        """
        if not 0.0 <= lambda_conf <= 1.0:
            raise ValueError("lambda_conf doit être compris entre 0 et 1.")
        if lambda_conf == 0:
            return {"total_properties": 0, "degraded_properties": [], "total_altered": 0}

        src_f = f":`{src_label}`" if src_label else ""
        dst_f = f":`{dst_label}`" if dst_label else ""
        if element_type == "node":
            match    = f"MATCH (x:`{label_or_type}`)"
            match_id = "MATCH (x) WHERE elementId(x) = id"
        else:
            match    = f"MATCH ({src_f})-[x:`{label_or_type}`]->({dst_f})"
            match_id = "MATCH ()-[x]->() WHERE elementId(x) = id"

        with self.driver.session(database=self.database) as session:
            type_rows = _property_samples(session, match)

            excluded = set(exclude_props or [])
            keys = sorted(
                r["prop"] for r in type_rows
                if r["prop"] not in excluded
                and not any(isinstance(v, (list, tuple)) for v in r["samples"])
            )

            total_altered = 0
            for key in keys:
                val_rows = session.run(f"""
                    {match}
                    WHERE x[$key] IS NOT NULL
                    RETURN elementId(x) AS eid, x[$key] AS val
                """, key=key).data()
                eids = sorted(r["eid"] for r in val_rows
                              if not isinstance(r["val"], (list, tuple))
                              and (scope_ids is None or r["eid"] in scope_ids))

                state_key = (element_type, label_or_type, src_label, dst_label, key)
                done = self._nonconform.setdefault(state_key, set())
                to_alter = int(lambda_conf * len(eids)) - len(done)
                if to_alter <= 0:
                    continue
                clean = [e for e in eids if e not in done]
                sample = self.rng.sample(clean, min(to_alter, len(clean)))
                done.update(sample)

                for i in range(0, len(sample), BATCH):
                    session.run(f"""
                        UNWIND $ids AS id
                        {match_id}
                        SET x[$key] = toString(x[$key]) + $suffix
                    """, ids=sample[i:i + BATCH], key=key, suffix=NONCONFORM_SUFFIX)
                total_altered += len(sample)

            return {
                "total_properties": len(keys),
                "degraded_properties": keys,
                "total_altered": total_altered,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Node-edge duplicates (uniqueness degradation)
    # ─────────────────────────────────────────────────────────────────────────

    def degrade_node_edge_duplicates(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        lambda_dup: float,
        scope_pairs: set = None,
    ) -> dict:
        """
        Duplicates a proportion lambda_dup of the (src_label)-[rel_type]->(dst_label)
        edges together with their source and destination nodes. Each copy is an
        isolated pattern (s')-[r']->(d'): s', r' and d' carry the labels and
        properties of s, r and d, but none of the other edges of s and d.
        A self-loop (s = d) is copied as one node with a self-loop.

        Cumulative target: only original edges are drawn, each at most once, so
        that the graph at level λ' contains all the copies of level λ.
        remove_duplicates() deletes every copy made by this degrader.
        scope_pairs: (src elementId, dst elementId) of the edges that may be
        duplicated (e.g. those of the training set); lambda_dup is then a
        proportion of them. None = every edge. The scope is read on the first
        call, when the graph is still original.
        """
        if not 0.0 <= lambda_dup <= 1.0:
            raise ValueError("lambda_dup doit être compris entre 0 et 1.")

        state_key = (src_label, rel_type, dst_label)
        rel_f = "`" + rel_type.replace("`", "``") + "`"

        with self.driver.session(database=self.database) as session:
            if state_key not in self._dup_pool:   # first call: edges are original
                rows = session.run(f"""
                    MATCH (s:`{src_label}`)-[r:`{rel_type}`]->(d:`{dst_label}`)
                    RETURN elementId(r) AS eid, elementId(s) AS s, elementId(d) AS d
                """).data()
                self._dup_pool[state_key] = sorted(
                    r["eid"] for r in rows
                    if scope_pairs is None or (r["s"], r["d"]) in scope_pairs)
            pool = self._dup_pool[state_key]
            done = self._dup_done.setdefault(state_key, set())

            to_dup = int(lambda_dup * len(pool)) - len(done)
            if to_dup <= 0:
                return {"total_edges": len(pool), "duplicated": 0,
                        "total_duplicated": len(done), "nodes_created": 0}

            remaining = [e for e in pool if e not in done]
            sample = self.rng.sample(remaining, min(to_dup, len(remaining)))
            done.update(sample)

            # Labels cannot be parameters in CREATE: the copies are grouped by
            # the label sets of the endpoints, one query per group (no APOC and
            # no dynamic labels needed).
            info = session.run("""
                UNWIND $eids AS eid
                MATCH (s)-[r]->(d) WHERE elementId(r) = eid
                RETURN eid, labels(s) AS sl, labels(d) AS dl, s = d AS loop
            """, eids=sample).data()
            groups: dict[tuple, list] = {}
            for r in info:
                k = (tuple(sorted(r["sl"])), tuple(sorted(r["dl"])), bool(r["loop"]))
                groups.setdefault(k, []).append(r["eid"])

            nodes_created = 0
            for (sl, dl, loop), eids in sorted(groups.items()):
                if loop:
                    query = f"""
                        UNWIND $eids AS eid
                        MATCH (s)-[r]->() WHERE elementId(r) = eid
                        CREATE (s2{_label_expr(sl)}) SET s2 = properties(s)
                        CREATE (s2)-[r2:{rel_f}]->(s2) SET r2 = properties(r)
                        RETURN elementId(s) AS s, elementId(s) AS d,
                               elementId(s2) AS s2, elementId(s2) AS d2
                    """
                else:
                    query = f"""
                        UNWIND $eids AS eid
                        MATCH (s)-[r]->(d) WHERE elementId(r) = eid
                        CREATE (s2{_label_expr(sl)}) SET s2 = properties(s)
                        CREATE (d2{_label_expr(dl)}) SET d2 = properties(d)
                        CREATE (s2)-[r2:{rel_f}]->(d2) SET r2 = properties(r)
                        RETURN elementId(s) AS s, elementId(d) AS d,
                               elementId(s2) AS s2, elementId(d2) AS d2
                    """
                for i in range(0, len(eids), BATCH):
                    for rec in session.run(query, eids=eids[i:i + BATCH]):
                        new = [rec["s2"]] if loop else [rec["s2"], rec["d2"]]
                        self._dup_nodes.extend(new)
                        nodes_created += len(new)
                        self.dup_origin_nodes[rec["s2"]] = rec["s"]
                        self.dup_origin_nodes[rec["d2"]] = rec["d"]
                        self.dup_origin_edges[(rec["s2"], rec["d2"])] = (rec["s"], rec["d"])

            return {
                "total_edges": len(pool),
                "duplicated": len(sample),
                "total_duplicated": len(done),
                "nodes_created": nodes_created,
            }

    def remove_duplicates(self) -> int:
        """Deletes the node copies (and their edges) made by
        degrade_node_edge_duplicates and forgets the duplication state."""
        ids = self._dup_nodes
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(ids), BATCH):
                session.run("""
                    UNWIND $ids AS id
                    MATCH (n) WHERE elementId(n) = id
                    DETACH DELETE n
                """, ids=ids[i:i + BATCH])
        removed = len(ids)
        self._dup_nodes = []
        self._dup_pool.clear()
        self._dup_done.clear()
        self.dup_origin_nodes.clear()
        self.dup_origin_edges.clear()
        return removed


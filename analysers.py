import json
import re
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from neo4j_pool import get_driver
from value_types import matches_regex
from statistics import mean
from typing import Literal

ElementType = Literal["node", "edge"]


class DataFreshnessAnalyser:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = get_driver(uri, user, password)
        self.database = database

    def close(self): pass  # shared driver, see neo4j_pool

    def _get_earliest_cursor(self, session):
        result = session.run("CALL db.cdc.earliest() YIELD id RETURN id")
        return result.single()["id"]

    @staticmethod
    def _selectors(element_type: ElementType, label_or_type: str,
                   src_label: str = None, dst_label: str = None) -> list[dict]:
        """CDC selectors: the server filters the change log by label/type
        instead of returning every event of the database."""
        if element_type == "node":
            return [{"select": "n", "labels": [label_or_type]}]
        sel = {"select": "r", "type": label_or_type}
        if src_label:
            sel["start"] = {"labels": [src_label]}
        if dst_label:
            sel["end"] = {"labels": [dst_label]}
        return [sel]

    @staticmethod
    def _match_alive(element_type: ElementType, label_or_type: str,
                     src_label: str = None, dst_label: str = None) -> str:
        """Keeps only elements that still exist (elementId seek, no label scan)."""
        if element_type == "node":
            return f"MATCH (c) WHERE elementId(c) = id AND c:{label_or_type}"
        src_filter = f" AND src:{src_label}" if src_label else ""
        dst_filter = f" AND dst:{dst_label}" if dst_label else ""
        return (f"MATCH (src)-[c]->(dst) WHERE elementId(c) = id "
                f"AND type(c) = '{label_or_type}'{src_filter}{dst_filter}")

    def _cdc_aggregate(self, element_type: ElementType, label_or_type: str,
                       src_label: str, dst_label: str,
                       event_filter: str, per_element: str, final: str,
                       **params):
        """
        Reads the CDC log for one type, aggregates per element, keeps the
        elements still present in the graph, then applies the final aggregate.
        """
        query = f"""
        CALL db.cdc.earliest() YIELD id AS cursor
        CALL db.cdc.query(cursor, $selectors)
        YIELD event, metadata
        {event_filter}
        WITH event.elementId AS id, {per_element}
        {self._match_alive(element_type, label_or_type, src_label, dst_label)}
        {final}
        """
        selectors = self._selectors(element_type, label_or_type, src_label, dst_label)
        with self.driver.session(database=self.database) as session:
            result = session.run(query, selectors=selectors, **params).single()
        return result["value"] if result else None

    def avg_age_of_type(self, element_type: ElementType, label_or_type: str,
                          src_label: str = None, dst_label: str = None):
        """Temps moyen depuis la dernière modification de chaque élément."""
        return self._cdc_aggregate(
            element_type, label_or_type, src_label, dst_label,
            event_filter="",
            per_element="max(metadata.txCommitTime) AS lastUpdate",
            final="RETURN avg(duration.inSeconds(lastUpdate, datetime()).seconds) AS value",
        )

    def update_frequency(self, element_type: ElementType, label_or_type: str,
                          src_label: str = None, dst_label: str = None,
                          days: int = 30) -> float:
        """
        Nombre moyen de mises à jour par instance sur une période donnée (en jours).
        Pour les arcs, src_label et dst_label permettent de filtrer par nœuds source/destination.
        """
        return self._cdc_aggregate(
            element_type, label_or_type, src_label, dst_label,
            event_filter="WHERE metadata.txCommitTime >= datetime() - duration({days: $days})",
            per_element="count(*) AS nb_updates",
            final="RETURN avg(nb_updates) AS value",
            days=int(days),
        )

    def avg_age_of_property(self, element_type: ElementType,
                             label_or_type: str, property_name: str,
                             src_label: str = None, dst_label: str = None):
        """Temps moyen écoulé depuis la dernière modification d'une propriété."""
        return self._cdc_aggregate(
            element_type, label_or_type, src_label, dst_label,
            event_filter=(f"WHERE event.state.after IS NOT NULL "
                          f"AND event.state.after.properties.{property_name} IS NOT NULL"),
            per_element="max(metadata.txCommitTime) AS lastUpdate",
            final="RETURN avg(duration.inSeconds(lastUpdate, datetime()).seconds) AS value",
        )


class DataConformityAnalyser:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = get_driver(uri, user, password)
        self.database = database

    def close(self): pass  # shared driver, see neo4j_pool

    def _conformity(self, element_type: ElementType, label_or_type: str,
                    property_name: str, regex: str, src_label: str = None,
                    dst_label: str = None, conditional: bool = False) -> float:
        """
        Share of the instances whose value fully matches the regex.

        Cypher's =~ only works on strings: it returns null on a Date, a
        DateTime or a number, which would then never conform. The regex is
        therefore tested here, in Python, on the text forms of each value
        (value_types.regex_texts), whatever its storage type: string as is,
        number as text, temporal value in ISO 8601. The query groups the
        instances by value, so each distinct value is tested once.

        conditional=False: missing values count as non-conforming.
        conditional=True : instances without the property are left out.
        """
        pattern = re.compile(regex)
        if element_type == "node":
            match = f"MATCH (x:`{label_or_type}`)"
        else:
            src_f = f":`{src_label}`" if src_label else ""
            dst_f = f":`{dst_label}`" if dst_label else ""
            match = f"MATCH ({src_f})-[x:`{label_or_type}`]->({dst_f})"
        total = conform = 0
        with self.driver.session(database=self.database) as session:
            for row in session.run(f"{match} RETURN x[$prop] AS v, count(*) AS c",
                                   prop=property_name):
                v, c = row["v"], row["c"]
                if v is None and conditional:
                    continue
                total += c
                if v is not None and matches_regex(v, pattern):
                    conform += c
        return 1.0 if total == 0 else conform / total

    def conformity_of_property(self, element_type: ElementType, label_or_type: str,
                                property_name: str,
                                regex: str = r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?", src_label: str = None, dst_label: str = None) -> float:
        """
        PropertyConformity : proportion de TOUTES les instances du type dont la
        valeur matche la regex ; une valeur absente compte comme non conforme.
        La valeur est testée quel que soit son type de stockage (voir _conformity).
        """
        return self._conformity(element_type, label_or_type, property_name, regex,
                                src_label, dst_label, conditional=False)

    def avg_conformity_of_type(self, element_type: ElementType, label_or_type: str,
                                properties: list[str], regexes: list[str]) -> float:
        scores = [
            self.conformity_of_property(element_type, label_or_type, prop, reg)
            for prop, reg in zip(properties, regexes)
        ]
        return mean(scores) if scores else 0.0


class DataCompletenessAnalyzer:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = get_driver(uri, user, password)
        self.database = database

    def close(self): pass  # shared driver, see neo4j_pool

    def completeness_of_property(self, element_type: ElementType,
                                  label_or_type: str, property_name: str, src_label: str = None, dst_label: str = None) -> float:
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                MATCH (e:{label_or_type})
                RETURN count(e) AS total, count(e.{property_name}) AS non_null
                """
            else:
                src_filter = f":{src_label}" if src_label else ""
                dst_filter = f":{dst_label}" if dst_label else ""
                query = f"""
                MATCH ({src_filter})-[e:{label_or_type}]->({dst_filter})
                RETURN count(e) AS total, count(e.{property_name}) AS non_null
                """
            result = session.run(query).single()
            if not result or result["total"] == 0:
                return 0.0
            return result["non_null"] / result["total"]

    def avg_completeness_of_type(self, element_type: ElementType,
                                 label_or_type: str, properties: list[str]) -> float:
        scores = [
            self.completeness_of_property(element_type, label_or_type, prop)
            for prop in properties
        ]
        return mean(scores) if scores else 0.0

    def completeness_of_node_edge(self, src_label: str, dst_label: str,
                                   rel_type: str) -> float:
        """
        Proportion de nœuds src_label ayant au moins une relation rel_type
        vers un nœud dst_label.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (src:{src_label})-[:{rel_type}]->(dst:{dst_label})
                RETURN count(DISTINCT src) AS with_edge
            """).single()
            with_edge = result["with_edge"] if result else 0

            result2 = session.run(f"""
                MATCH (n:{src_label}) RETURN count(n) AS total
            """).single()
            total = result2["total"] if result2 else 0

            return (with_edge / total) if total > 0 else 0.0

    def weighted_completeness_of_type(self, element_type: ElementType,
                                       label_or_type: str,
                                       properties: list[str],
                                       weights: list[float],
                                       src_label: str = None,
                                       dst_label: str = None) -> float:
        """
        WeightedTypeCompleteness : moyenne pondérée de PropertyCompleteness.
        sum(weights) doit être égal à 1.
        """
        scores = [
            self.completeness_of_property(element_type, label_or_type, prop, src_label=src_label, dst_label=dst_label)
            for prop in properties
        ]
        return sum(w * s for w, s in zip(weights, scores))


class DataConformityAnalyserExt(DataConformityAnalyser):
    """Extensions : ConditionalPropertyConformity, WeightedTypeConformity."""

    def conditional_conformity_of_property(self, element_type: ElementType,
                                            label_or_type: str, property_name: str,
                                            regex: str,
                                            src_label: str = None,
                                            dst_label: str = None) -> float:
        """
        ConditionalPropertyConformity : conformité uniquement sur les instances
        ayant la propriété renseignée (exclut les null du dénominateur).
        La valeur est testée quel que soit son type de stockage (voir _conformity).
        """
        return self._conformity(element_type, label_or_type, property_name, regex,
                                src_label, dst_label, conditional=True)

    def weighted_conformity_of_type(self, element_type: ElementType,
                                     label_or_type: str,
                                     properties: list[str],
                                     regexes: list[str],
                                     weights: list[float],
                                     src_label: str = None,
                                     dst_label: str = None) -> float:
        """
        WeightedTypeConformity : sum(w_p * PropertyConformity(p)).
        sum(weights) doit être égal à 1.
        """
        scores = [
            self.conformity_of_property(element_type, label_or_type, prop, reg, src_label=src_label, dst_label=dst_label)
            for prop, reg in zip(properties, regexes)
        ]
        return sum(w * s for w, s in zip(weights, scores))


class DataFreshnessAnalyserExt(DataFreshnessAnalyser):
    """Extensions : FreshnessScore, UpdateDelay."""

    def freshness_score(self, element_type: ElementType,
                        label_or_type: str, property_name: str,
                        src_label: str = None, dst_label: str = None) -> float:
        """
        FreshnessScore(p,t) = 1 / (1 + PropertyAge(p,t))
        Retourne une valeur entre 0 et 1.
        """
        age = self.avg_age_of_property(element_type, label_or_type, property_name,
                                        src_label=src_label, dst_label=dst_label)
        if age is None:
            return None
        return 1.0 / (1.0 + age)

    def update_delay(self, element_type: ElementType, label_or_type: str,
                     src_label: str = None, dst_label: str = None) -> float:
        """
        UpdateDelay : temps moyen entre deux mises à jour successives (en secondes).
        """
        return self._cdc_aggregate(
            element_type, label_or_type, src_label, dst_label,
            event_filter="",
            per_element="collect(metadata.txCommitTime) AS times WHERE size(times) > 1",
            final="""WITH [i IN range(0, size(times)-2) |
                    duration.inSeconds(times[i], times[i+1]).seconds] AS gaps
        UNWIND gaps AS gap
        RETURN avg(gap) AS value""",
        )


class DataUniquenessAnalyser:
    """
    Métriques d'unicité :
    - exact_duplicate_rate   : ExactDuplicateRate(t)
    - partial_duplicate_rate : PartialDuplicateRate(t, θ)
    - uniqueness_score       : UniquenessScore(t) = 1 - ExactDuplicateRate(t)
    - average_redundancy     : AverageRedundancy(t)
    """

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = get_driver(uri, user, password)
        self.database = database

    def close(self): pass  # shared driver, see neo4j_pool

    def _fetch_instances(self, session, element_type: ElementType,
                         label_or_type: str,
                         src_label: str = None, dst_label: str = None) -> list[dict]:
        """Récupère les propriétés de toutes les instances du type."""
        if element_type == "node":
            query = f"MATCH (n:{label_or_type}) RETURN properties(n) AS props"
        else:
            src_f = f":{src_label}" if src_label else ""
            dst_f = f":{dst_label}" if dst_label else ""
            query = f"MATCH ({src_f})-[e:{label_or_type}]->({dst_f}) RETURN properties(e) AS props"
        return [dict(r["props"]) for r in session.run(query)]

    @staticmethod
    def _sim(x: dict, y: dict) -> float:
        """sim(x,y) = |clés communes avec même valeur| / |union des clés|"""
        all_keys = set(x) | set(y)
        if not all_keys:
            return 1.0
        common_equal = sum(1 for k in all_keys if k in x and k in y and x[k] == y[k])
        return common_equal / len(all_keys)

    @staticmethod
    def _pair_stats(instances: list[dict], threshold: float = None,
                    pairwise: bool = True) -> tuple[float, float, float, float]:
        """
        Same result as applying _sim to every pair, without the O(n²) Python loop.

        1. Identical instances are grouped: pairs inside a group have sim = 1.
        2. Between distinct groups, sim is computed by blocks of sparse products:
             common_equal(x, y)  = A·Aᵀ                 A: one-hot of (key, value)
             |keys_x ∪ keys_y|   = |kx| + |ky| - K·Kᵀ   K: one-hot of key
        Returns (total_pairs, exact_pairs, partial_pairs, sum_sim).
        """
        n = len(instances)
        total = n * (n - 1) / 2

        def _v(value):
            return json.dumps(value, sort_keys=True, default=str)

        groups: dict[str, dict] = {}
        counter: Counter = Counter()
        for x in instances:
            sig = _v(x)
            groups.setdefault(sig, x)
            counter[sig] += 1
        reps   = list(groups.values())
        counts = np.array([counter[s] for s in groups], dtype=float)

        within  = float((counts * (counts - 1) / 2).sum())
        exact   = within
        partial = within if (threshold is None or threshold <= 1.0) else 0.0
        sum_sim = within

        g = len(reps)
        if not pairwise or g < 2:
            return total, exact, partial, sum_sim

        item_ids: dict = {}
        key_ids: dict = {}
        rows, cols_a, cols_k = [], [], []
        for i, x in enumerate(reps):
            for k, v in x.items():
                rows.append(i)
                cols_k.append(key_ids.setdefault(k, len(key_ids)))
                cols_a.append(item_ids.setdefault((k, _v(v)), len(item_ids)))
        ones = np.ones(len(rows))
        A = csr_matrix((ones, (rows, cols_a)), shape=(g, max(1, len(item_ids))))
        K = csr_matrix((ones, (rows, cols_k)), shape=(g, max(1, len(key_ids))))
        n_keys = np.asarray(K.sum(axis=1)).ravel()
        At, Kt = A.T.tocsc(), K.T.tocsc()

        block = max(1, 2_000_000 // g)
        cols = np.arange(g)
        for start in range(0, g, block):
            stop = min(start + block, g)
            common = (A[start:stop] @ At).toarray()
            shared = (K[start:stop] @ Kt).toarray()
            union  = n_keys[start:stop, None] + n_keys[None, :] - shared
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, common / union, 1.0)
            upper = cols[None, :] > np.arange(start, stop)[:, None]   # pairs i < j
            s = sim[upper]
            w = (counts[start:stop, None] * counts[None, :])[upper]
            sum_sim += float((s * w).sum())
            if threshold is not None:
                partial += float(w[s >= threshold].sum())
        return total, exact, partial, sum_sim

    def exact_duplicate_rate(self, element_type: ElementType,
                              label_or_type: str,
                              src_label: str = None, dst_label: str = None) -> float:
        """ExactDuplicateRate(t) : proportion de paires exactement identiques."""
        with self.driver.session(database=self.database) as session:
            instances = self._fetch_instances(session, element_type, label_or_type,
                                              src_label, dst_label)
        if len(instances) < 2:
            return 0.0
        total, exact, _, _ = self._pair_stats(instances, pairwise=False)
        return exact / total

    def partial_duplicate_rate(self, element_type: ElementType,
                                label_or_type: str, threshold: float = 0.8,
                                src_label: str = None, dst_label: str = None) -> float:
        """PartialDuplicateRate(t, θ) : proportion de paires avec sim >= θ."""
        with self.driver.session(database=self.database) as session:
            instances = self._fetch_instances(session, element_type, label_or_type,
                                              src_label, dst_label)
        if len(instances) < 2:
            return 0.0
        total, _, partial, _ = self._pair_stats(instances, threshold=threshold)
        return partial / total

    def uniqueness_score(self, element_type: ElementType,
                          label_or_type: str,
                          src_label: str = None, dst_label: str = None) -> float:
        """UniquenessScore(t) = 1 - ExactDuplicateRate(t)."""
        return 1.0 - self.exact_duplicate_rate(element_type, label_or_type,
                                                src_label, dst_label)

    def average_redundancy(self, element_type: ElementType,
                            label_or_type: str,
                            src_label: str = None, dst_label: str = None) -> float:
        """AverageRedundancy(t) : similarité moyenne sur toutes les paires."""
        with self.driver.session(database=self.database) as session:
            instances = self._fetch_instances(session, element_type, label_or_type,
                                              src_label, dst_label)
        if len(instances) < 2:
            return 0.0
        total, _, _, sum_sim = self._pair_stats(instances)
        return sum_sim / total


class DataAccuracyAnalyser:
    """
    Métriques d'exactitude :
    - property_accuracy : PropertyAccuracy(p, t, v_ref)
    - type_accuracy     : TypeAccuracy(t, P, refs)
    """

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = get_driver(uri, user, password)
        self.database = database

    def close(self): pass  # shared driver, see neo4j_pool

    def property_accuracy(self, element_type: ElementType,
                           label_or_type: str, property_name: str,
                           reference_value: float,
                           src_label: str = None, dst_label: str = None) -> float:
        """
        PropertyAccuracy(p, t) = mean(|π(x,p) - v_ref|) sur les instances.
        Seules les instances avec une valeur numérique sont considérées (les
        nombres stockés en chaîne comptent) ; dates, textes, booléens et NaN
        sont ignorés au lieu de faire échouer toFloat.
        Retourne None si aucune instance n'a de valeur numérique.
        """
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                match = f"MATCH (x:`{label_or_type}`)"
            else:
                src_f = f":`{src_label}`" if src_label else ""
                dst_f = f":`{dst_label}`" if dst_label else ""
                match = f"MATCH ({src_f})-[x:`{label_or_type}`]->({dst_f})"
            query = f"""
                {match}
                WITH toFloatOrNull(x[$prop]) AS v
                WHERE v IS NOT NULL AND v = v
                RETURN avg(abs(v - $ref)) AS avg_error, count(v) AS n
            """
            result = session.run(query, prop=property_name,
                                 ref=float(reference_value)).single()
            if not result or result["n"] == 0:
                return None
            return result["avg_error"]

    def type_accuracy(self, element_type: ElementType,
                      label_or_type: str,
                      properties: list[str],
                      reference_values: list[float],
                      src_label: str = None, dst_label: str = None) -> float:
        """
        TypeAccuracy(t, P) = mean(PropertyAccuracy(p, t)) sur toutes les propriétés.
        """
        values = (
            self.property_accuracy(element_type, label_or_type, prop, ref,
                                   src_label, dst_label)
            for prop, ref in zip(properties, reference_values)
        )
        scores = [v for v in values if v is not None]
        return mean(scores) if scores else None

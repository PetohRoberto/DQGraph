from neo4j import GraphDatabase
from statistics import mean
from typing import Literal

ElementType = Literal["node", "edge"]


class DataFreshnessAnalyser:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self): self.driver.close()

    def _get_earliest_cursor(self, session):
        result = session.run("CALL db.cdc.earliest() YIELD id RETURN id")
        return result.single()["id"]

    def avg_age_of_type(self, element_type: ElementType, label_or_type: str,
                          src_label: str = None, dst_label: str = None):
        """Temps moyen depuis la dernière modification de chaque élément."""
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (c:{label_or_type})
                WHERE elementId(c) = event.elementId
                WITH event.elementId AS id, max(metadata.txCommitTime) AS lastUpdate
                WITH id, duration.inSeconds(lastUpdate, datetime()).seconds AS age
                RETURN avg(age) AS avg_age
                """
                result = session.run(query).single()
            else:
                src_filter = f":{src_label}" if src_label else ""
                dst_filter = f":{dst_label}" if dst_label else ""
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (src{src_filter})-[c:{label_or_type}]->(dst{dst_filter})
                WHERE elementId(c) = event.elementId
                WITH event.elementId AS id, max(metadata.txCommitTime) AS lastUpdate
                WITH id, duration.inSeconds(lastUpdate, datetime()).seconds AS age
                RETURN avg(age) AS avg_age
                """
                result = session.run(query).single()
            return result["avg_age"] if result else None

    def update_frequency(self, element_type: ElementType, label_or_type: str,
                          src_label: str = None, dst_label: str = None,
                          days: int = 30) -> float:
        """
        Nombre moyen de mises à jour par instance sur une période donnée (en jours).
        Pour les arcs, src_label et dst_label permettent de filtrer par nœuds source/destination.
        """
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (c:{label_or_type})
                WHERE elementId(c) = event.elementId
                  AND metadata.txCommitTime >= datetime() - duration({{days: {days}}})
                WITH event.elementId AS id, count(*) AS nb_updates
                RETURN avg(nb_updates) AS avg_updates
                """
                result = session.run(query).single()
            else:
                src_filter = f":{src_label}" if src_label else ""
                dst_filter = f":{dst_label}" if dst_label else ""
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (src{src_filter})-[c:{label_or_type}]->(dst{dst_filter})
                WHERE elementId(c) = event.elementId
                  AND metadata.txCommitTime >= datetime() - duration({{days: {days}}})
                WITH event.elementId AS id, count(*) AS nb_updates
                RETURN avg(nb_updates) AS avg_updates
                """
                result = session.run(query).single()
            return result["avg_updates"] if result else None

    def avg_age_of_property(self, element_type: ElementType,
                             label_or_type: str, property_name: str,
                             src_label: str = None, dst_label: str = None):
        """Temps moyen écoulé depuis la dernière modification d'une propriété."""
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (c:{label_or_type})
                WHERE elementId(c) = event.elementId
                  AND event.state.after IS NOT NULL
                  AND event.state.after.properties.{property_name} IS NOT NULL
                WITH event.elementId AS id, max(metadata.txCommitTime) AS lastUpdate
                WITH id, duration.inSeconds(lastUpdate, datetime()).seconds AS age
                RETURN avg(age) AS avg_age
                """
                result = session.run(query).single()
            else:
                src_filter = f":{src_label}" if src_label else ""
                dst_filter = f":{dst_label}" if dst_label else ""
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (src{src_filter})-[c:{label_or_type}]->(dst{dst_filter})
                WHERE elementId(c) = event.elementId
                  AND event.state.after IS NOT NULL
                  AND event.state.after.properties.{property_name} IS NOT NULL
                WITH event.elementId AS id, max(metadata.txCommitTime) AS lastUpdate
                WITH id, duration.inSeconds(lastUpdate, datetime()).seconds AS age
                RETURN avg(age) AS avg_age
                """
                result = session.run(query).single()
            return result["avg_age"] if result else None


class DataConformityAnalyser:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self): self.driver.close()

    def conformity_of_property(self, element_type: ElementType, label_or_type: str,
                                property_name: str,
                                regex: str = r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?", src_label: str = None, dst_label: str = None) -> float:
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                MATCH (n:{label_or_type})
                WHERE n.{property_name} IS NOT NULL
                RETURN count(n.{property_name}) AS total,
                       count(CASE WHEN n.{property_name} =~ $regex THEN 1 END) AS conform
                """
            else:
                src_filter = f":{src_label}" if src_label else ""
                dst_filter = f":{dst_label}" if dst_label else ""
                query = f"""
                MATCH ({src_filter})-[e:{label_or_type}]-({dst_filter})
                WHERE e.{property_name} IS NOT NULL
                RETURN count(e.{property_name}) AS total,
                       count(CASE WHEN e.{property_name} =~ $regex THEN 1 END) AS conform
                """
            result = session.run(query, regex=regex).single()
            if not result or result["total"] == 0:
                return 1.0
            return result["conform"] / result["total"]

    def avg_conformity_of_type(self, element_type: ElementType, label_or_type: str,
                                properties: list[str], regexes: list[str]) -> float:
        scores = [
            self.conformity_of_property(element_type, label_or_type, prop, reg)
            for prop, reg in zip(properties, regexes)
        ]
        return mean(scores) if scores else 0.0


class DataCompletenessAnalyzer:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self): self.driver.close()

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
                MATCH ({src_filter})-[e:{label_or_type}]-({dst_filter})
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
        """
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                MATCH (n:{label_or_type})
                WHERE n.{property_name} IS NOT NULL
                RETURN count(n) AS total,
                       count(CASE WHEN n.{property_name} =~ $regex THEN 1 END) AS conform
                """
            else:
                src_f = f":{src_label}" if src_label else ""
                dst_f = f":{dst_label}" if dst_label else ""
                query = f"""
                MATCH ({src_f})-[e:{label_or_type}]-({dst_f})
                WHERE e.{property_name} IS NOT NULL
                RETURN count(e) AS total,
                       count(CASE WHEN e.{property_name} =~ $regex THEN 1 END) AS conform
                """
            result = session.run(query, regex=regex).single()
            if not result or result["total"] == 0:
                return 1.0
            return result["conform"] / result["total"]

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
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (c:{label_or_type})
                WHERE elementId(c) = event.elementId
                WITH event.elementId AS id,
                     collect(metadata.txCommitTime) AS times
                WHERE size(times) > 1
                WITH id, [i IN range(0, size(times)-2) |
                    duration.inSeconds(times[i], times[i+1]).seconds] AS gaps
                UNWIND gaps AS gap
                RETURN avg(gap) AS avg_delay
                """
            else:
                src_f = f":{src_label}" if src_label else ""
                dst_f = f":{dst_label}" if dst_label else ""
                query = f"""
                CALL db.cdc.earliest() YIELD id AS cursor
                CALL db.cdc.query(cursor)
                YIELD event, metadata
                MATCH (src{src_f})-[c:{label_or_type}]->(dst{dst_f})
                WHERE elementId(c) = event.elementId
                WITH event.elementId AS id,
                     collect(metadata.txCommitTime) AS times
                WHERE size(times) > 1
                WITH id, [i IN range(0, size(times)-2) |
                    duration.inSeconds(times[i], times[i+1]).seconds] AS gaps
                UNWIND gaps AS gap
                RETURN avg(gap) AS avg_delay
                """
            result = session.run(query).single()
            return result["avg_delay"] if result else None


class DataUniquenessAnalyser:
    """
    Métriques d'unicité :
    - exact_duplicate_rate   : ExactDuplicateRate(t)
    - partial_duplicate_rate : PartialDuplicateRate(t, θ)
    - uniqueness_score       : UniquenessScore(t) = 1 - ExactDuplicateRate(t)
    - average_redundancy     : AverageRedundancy(t)
    """

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self): self.driver.close()

    def _fetch_instances(self, session, element_type: ElementType,
                         label_or_type: str,
                         src_label: str = None, dst_label: str = None) -> list[dict]:
        """Récupère les propriétés de toutes les instances du type."""
        if element_type == "node":
            query = f"MATCH (n:{label_or_type}) RETURN properties(n) AS props"
        else:
            src_f = f":{src_label}" if src_label else ""
            dst_f = f":{dst_label}" if dst_label else ""
            query = f"MATCH ({src_f})-[e:{label_or_type}]-({dst_f}) RETURN properties(e) AS props"
        return [dict(r["props"]) for r in session.run(query)]

    @staticmethod
    def _sim(x: dict, y: dict) -> float:
        """sim(x,y) = |clés communes avec même valeur| / |union des clés|"""
        all_keys = set(x) | set(y)
        if not all_keys:
            return 1.0
        common_equal = sum(1 for k in all_keys if k in x and k in y and x[k] == y[k])
        return common_equal / len(all_keys)

    def exact_duplicate_rate(self, element_type: ElementType,
                              label_or_type: str,
                              src_label: str = None, dst_label: str = None) -> float:
        """ExactDuplicateRate(t) : proportion de paires exactement identiques."""
        with self.driver.session(database=self.database) as session:
            instances = self._fetch_instances(session, element_type, label_or_type,
                                              src_label, dst_label)
        n = len(instances)
        if n < 2:
            return 0.0
        total_pairs = n * (n - 1) / 2
        exact = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if self._sim(instances[i], instances[j]) == 1.0
        )
        return exact / total_pairs

    def partial_duplicate_rate(self, element_type: ElementType,
                                label_or_type: str, threshold: float = 0.8,
                                src_label: str = None, dst_label: str = None) -> float:
        """PartialDuplicateRate(t, θ) : proportion de paires avec sim >= θ."""
        with self.driver.session(database=self.database) as session:
            instances = self._fetch_instances(session, element_type, label_or_type,
                                              src_label, dst_label)
        n = len(instances)
        if n < 2:
            return 0.0
        total_pairs = n * (n - 1) / 2
        partial = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if self._sim(instances[i], instances[j]) >= threshold
        )
        return partial / total_pairs

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
        n = len(instances)
        if n < 2:
            return 0.0
        total_pairs = n * (n - 1) / 2
        total_sim = sum(
            self._sim(instances[i], instances[j])
            for i in range(n) for j in range(i + 1, n)
        )
        return total_sim / total_pairs


class DataAccuracyAnalyser:
    """
    Métriques d'exactitude :
    - property_accuracy : PropertyAccuracy(p, t, v_ref)
    - type_accuracy     : TypeAccuracy(t, P, refs)
    """

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self): self.driver.close()

    def property_accuracy(self, element_type: ElementType,
                           label_or_type: str, property_name: str,
                           reference_value: float,
                           src_label: str = None, dst_label: str = None) -> float:
        """
        PropertyAccuracy(p, t) = mean(|π(x,p) - v_ref|) sur les instances.
        Seules les instances avec la propriété renseignée sont considérées.
        """
        with self.driver.session(database=self.database) as session:
            if element_type == "node":
                query = f"""
                MATCH (n:{label_or_type})
                WHERE n.{property_name} IS NOT NULL
                RETURN avg(abs(toFloat(n.{property_name}) - $ref)) AS avg_error
                """
            else:
                src_f = f":{src_label}" if src_label else ""
                dst_f = f":{dst_label}" if dst_label else ""
                query = f"""
                MATCH ({src_f})-[e:{label_or_type}]-({dst_f})
                WHERE e.{property_name} IS NOT NULL
                RETURN avg(abs(toFloat(e.{property_name}) - $ref)) AS avg_error
                """
            result = session.run(query, ref=float(reference_value)).single()
            return result["avg_error"] if result else None

    def type_accuracy(self, element_type: ElementType,
                      label_or_type: str,
                      properties: list[str],
                      reference_values: list[float],
                      src_label: str = None, dst_label: str = None) -> float:
        """
        TypeAccuracy(t, P) = mean(PropertyAccuracy(p, t)) sur toutes les propriétés.
        """
        scores = [
            self.property_accuracy(element_type, label_or_type, prop, ref,
                                   src_label, dst_label)
            for prop, ref in zip(properties, reference_values)
            if self.property_accuracy(element_type, label_or_type, prop, ref,
                                      src_label, dst_label) is not None
        ]
        return mean(scores) if scores else None

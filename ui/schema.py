from neo4j_pool import get_driver
from value_types import classify, NUMERIC_TYPES, CATEGORICAL_TYPES

# Values sampled per property key to infer its type (value_types.classify).
TYPE_SAMPLE = 1000
CLASSIFY_SAMPLES = 50


def _property_keys(session, match: str) -> list[str]:
    """Every property key of the elements `x` matched by `match`. Full scan on
    purpose: a key carried by a few instances only (a sparse property, shown as
    null elsewhere by Neo4j Browser) must not be missed by a sample."""
    return [r["k"] for r in session.run(
        f"{match} UNWIND keys(x) AS k RETURN DISTINCT k ORDER BY k")]


def _property_types(session, match: str) -> dict:
    """{key: type} for every property key of the elements `x` matched by
    `match`, from up to TYPE_SAMPLE non-null values of each key."""
    out = {}
    for k in _property_keys(session, match):
        r = session.run(f"""
            {match}
            WHERE x[$k] IS NOT NULL
            WITH x[$k] AS val LIMIT $n
            RETURN collect(val)[..$m] AS samples, count(DISTINCT val) AS distinct_count
        """, k=k, n=TYPE_SAMPLE, m=CLASSIFY_SAMPLES).single()
        out[k] = classify(r["samples"], r["distinct_count"])
    return out


class SchemaExplorer:
    """Récupère dynamiquement le schéma de la base Neo4j."""

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = get_driver(uri, user, password)
        self.database = database

    def close(self):
        pass  # shared driver, see neo4j_pool

    def get_node_labels(self) -> list[str]:
        with self.driver.session(database=self.database) as session:
            result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            return [r["label"] for r in result]

    def get_relationship_types(self) -> list[str]:
        with self.driver.session(database=self.database) as session:
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            return [r["relationshipType"] for r in result]

    def get_node_properties(self, label: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            return _property_keys(session, f"MATCH (x:`{label}`)")

    def get_edge_properties(self, rel_type: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            return _property_keys(session, f"MATCH ()-[x:`{rel_type}`]->()")

    # The schema helpers below only use patterns answered by Neo4j's counts
    # store — (:A)-[:T]->() and ()-[:T]->(:B) — which are O(1), instead of
    # scanning every relationship of the graph.

    def _count_out(self, session, label: str, rel_type: str) -> int:
        return session.run(
            f"MATCH (:`{label}`)-[r:`{rel_type}`]->() RETURN count(r) AS c"
        ).single()["c"]

    def _count_in(self, session, rel_type: str, label: str) -> int:
        return session.run(
            f"MATCH ()-[r:`{rel_type}`]->(:`{label}`) RETURN count(r) AS c"
        ).single()["c"]

    def get_node_labels_for_rel(self, rel_type: str) -> tuple[list[str], list[str]]:
        """Retourne (labels_source, labels_destination) pour un type de relation."""
        labels = self.get_node_labels()
        with self.driver.session(database=self.database) as session:
            sources = [l for l in labels if self._count_out(session, l, rel_type) > 0]
            destinations = [l for l in labels if self._count_in(session, rel_type, l) > 0]
        return sources, destinations


    def get_rel_types_for_source(self, src_label: str) -> list[str]:
        """Relationship types where src_label appears as source."""
        rel_types = self.get_relationship_types()
        with self.driver.session(database=self.database) as session:
            return [t for t in rel_types if self._count_out(session, src_label, t) > 0]

    def get_dst_labels_for(self, src_label: str, rel_type: str) -> list[str]:
        """Destination labels for a given source label and rel type."""
        labels = self.get_node_labels()
        out = []
        with self.driver.session(database=self.database) as session:
            for lbl in labels:
                # cheap pre-filter, then an existence check that stops at the first match
                if self._count_in(session, rel_type, lbl) == 0:
                    continue
                hit = session.run(
                    f"MATCH (:`{src_label}`)-[:`{rel_type}`]->(:`{lbl}`) "
                    "RETURN 1 AS x LIMIT 1"
                ).single()
                if hit:
                    out.append(lbl)
        return out

    def get_src_labels_for_rel(self, rel_type: str) -> list[str]:
        """Source labels for a given relationship type."""
        return self.get_node_labels_for_rel(rel_type)[0]

    def get_property_types(self, label: str) -> dict:
        """
        Returns {prop_name: type_str} for a node label (see value_types.classify):
        'numeric', 'numeric_discrete', 'temporal', 'categorical', 'string',
        'boolean', 'other'. Numbers and dates stored as strings are recognised.
        """
        with self.driver.session(database=self.database) as session:
            return _property_types(session, f"MATCH (x:`{label}`)")

    def get_edge_property_types(self, rel_type: str) -> dict:
        """Same as get_property_types, for the properties of a relationship type."""
        with self.driver.session(database=self.database) as session:
            return _property_types(session, f"MATCH ()-[x:`{rel_type}`]->()")

    def get_numeric_properties(self, label: str) -> list[str]:
        """Retourne les propriétés numériques (ou convertibles) d'un label."""
        types = self.get_property_types(label)
        return sorted([p for p, t in types.items() if t in NUMERIC_TYPES])

    def get_categorical_properties(self, label: str) -> list[str]:
        """Retourne les propriétés catégorielles d'un label."""
        types = self.get_property_types(label)
        return sorted([p for p, t in types.items() if t in CATEGORICAL_TYPES])

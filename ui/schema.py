from neo4j import GraphDatabase


class SchemaExplorer:
    """Récupère dynamiquement le schéma de la base Neo4j."""

    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

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
            result = session.run(f"""
                MATCH (n:{label})
                WITH n LIMIT 100
                UNWIND keys(n) AS prop
                RETURN DISTINCT prop ORDER BY prop
            """)
            return [r["prop"] for r in result]

    def get_edge_properties(self, rel_type: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH ()-[e:{rel_type}]-()
                WITH e LIMIT 100
                UNWIND keys(e) AS prop
                RETURN DISTINCT prop ORDER BY prop
            """)
            return [r["prop"] for r in result]

    def get_node_labels_for_rel(self, rel_type: str) -> tuple[list[str], list[str]]:
        """Retourne (labels_source, labels_destination) pour un type de relation."""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (src)-[:{rel_type}]->(dst)
                WITH DISTINCT labels(src) AS src_labels, labels(dst) AS dst_labels
                LIMIT 50
                RETURN src_labels, dst_labels
            """)
            sources, destinations = set(), set()
            for r in result:
                for l in r["src_labels"]: sources.add(l)
                for l in r["dst_labels"]: destinations.add(l)
            return sorted(sources), sorted(destinations)


    def get_rel_types_for_source(self, src_label: str) -> list[str]:
        """Relationship types where src_label appears as source."""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (:{src_label})-[r]->()
                RETURN DISTINCT type(r) AS rel ORDER BY rel
            """)
            return [r["rel"] for r in result]

    def get_dst_labels_for(self, src_label: str, rel_type: str) -> list[str]:
        """Destination labels for a given source label and rel type."""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (:{src_label})-[:{rel_type}]->(dst)
                UNWIND labels(dst) AS lbl
                RETURN DISTINCT lbl ORDER BY lbl LIMIT 50
            """)
            return [r["lbl"] for r in result]

    def get_src_labels_for_rel(self, rel_type: str) -> list[str]:
        """Source labels for a given relationship type."""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (src)-[:{rel_type}]->()
                UNWIND labels(src) AS lbl
                RETURN DISTINCT lbl ORDER BY lbl LIMIT 50
            """)
            return [r["lbl"] for r in result]

    def get_property_types(self, label: str) -> dict:
        """
        Returns {prop_name: type_str} for a node label.
        Types: 'numeric', 'categorical', 'string', 'boolean', 'other'
        Samples multiple non-null values to handle mixed types and NaN.
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(f"""
                MATCH (n:{label})
                WITH n LIMIT 500
                UNWIND keys(n) AS k
                WITH k, n[k] AS val
                WHERE val IS NOT NULL
                WITH k, collect(val)[..10] AS samples,
                     count(DISTINCT val) AS distinct_count
                RETURN k AS prop, samples, distinct_count
            """).data()

            types = {}
            for r in rows:
                prop    = r["prop"]
                samples = [s for s in r["samples"] if s is not None]
                ndist   = r["distinct_count"]

                if not samples:
                    types[prop] = "other"
                    continue

                # Check majority type from multiple samples
                num_count = sum(1 for s in samples
                                if isinstance(s, (int, float)) and not isinstance(s, bool)
                                and str(s) != "nan")
                bool_count = sum(1 for s in samples if isinstance(s, bool))
                str_count  = sum(1 for s in samples if isinstance(s, str))

                if bool_count > len(samples) / 2:
                    types[prop] = "boolean"
                elif num_count > len(samples) / 2:
                    types[prop] = "numeric"
                elif str_count > len(samples) / 2:
                    types[prop] = "categorical" if ndist <= 50 else "string"
                else:
                    types[prop] = "other"

            return types

    def get_numeric_properties(self, label: str) -> list[str]:
        """Retourne les propriétés numériques d'un label."""
        types = self.get_property_types(label)
        return sorted([p for p, t in types.items() if t == "numeric"])

    def get_categorical_properties(self, label: str) -> list[str]:
        """Retourne les propriétés catégorielles d'un label."""
        types = self.get_property_types(label)
        return sorted([p for p, t in types.items() if t in ("categorical", "boolean")])

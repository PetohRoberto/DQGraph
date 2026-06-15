import random
from neo4j import GraphDatabase
from typing import Literal

ElementType = Literal["node", "edge"]


class DataCompletenessDegrader:
    """
    Méthodes de dégradation pour les métriques de complétude.
    Toutes les opérations sont réversibles si l'on conserve les valeurs originales.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

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
            rows = session.run(ids_query).data()

            # 4. Tirage aléatoire
            sample = random.sample(rows, min(to_degrade, len(rows)))

            # 5. Apply degradation: REMOVE the property (null = missing)
            #    This avoids NaN issues with GDS and Neo4j.
            degraded_count = 0
            for row in sample:
                eid = row["eid"]

                if element_type == "node":
                    upd_query = f"""
                    MATCH (n:{label_or_type})
                    WHERE elementId(n) = $eid
                    REMOVE n.{property_name}
                    """
                else:
                    upd_query = f"""
                    MATCH ({src_f})-[e:{label_or_type}]->({dst_f})
                    WHERE elementId(e) = $eid
                    REMOVE e.{property_name}
                    """
                session.run(upd_query, eid=eid)
                degraded_count += 1

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
    ) -> dict:
        """
        Dégrade EdgeCoverage(t1, t2) jusqu'à atteindre un taux de nœuds sans arc
        de lambda_ec.

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

            # 3. Récupérer les nœuds source qui ont encore au moins un arc
            rows = session.run(f"""
                MATCH (src:{src_label})-[:{rel_type}]->(:{dst_label})
                RETURN DISTINCT elementId(src) AS eid
            """).data()

            sample = random.sample(rows, min(to_degrade, len(rows)))
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
        import random as _rnd

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
            rows = session.run(query).data()

            if not rows:
                return {"total": 0, "corrupted": 0, "lambda_num": lambda_num}

            # 2. Add Gaussian noise to each value
            corrupted = 0
            for row in rows:
                original = row["val"]
                if original is None:
                    continue
                noise = _rnd.gauss(0, lambda_num)
                new_val = original + noise

                if element_type == "node":
                    upd = f"""
                    MATCH (n:{label_or_type})
                    WHERE elementId(n) = $eid
                    SET n.{property_name} = $val
                    """
                else:
                    upd = f"""
                    MATCH ({src_f})-[e:{label_or_type}]->({dst_f})
                    WHERE elementId(e) = $eid
                    SET e.{property_name} = $val
                    """
                session.run(upd, eid=row["eid"], val=new_val)
                corrupted += 1

            return {
                "total": len(rows),
                "corrupted": corrupted,
                "lambda_num": lambda_num,
            }


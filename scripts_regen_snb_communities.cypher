// Regenerate KNOWS relationships on the existing Person nodes with an
// embedded community structure, so link prediction has real signal to learn.
//
// Run in Neo4j Browser / cypher-shell against the snb_sf database.
// Adjust :Person label / KNOWS type / community count if yours differ.

// 1. Tag each Person with a community (10 communities ~ 300 people each)
MATCH (p:Person)
WITH collect(p) AS people
UNWIND range(0, size(people) - 1) AS i
WITH people[i] AS p, i % 10 AS community
SET p.community = community;

// 2. Drop existing KNOWS edges (back up first if you need the originals!)
MATCH ()-[r:KNOWS]->()
DELETE r;

// 3. Re-create ~9000 KNOWS edges with 85% intra-community bias
MATCH (p:Person)
WITH collect(p) AS people
UNWIND range(1, 9000) AS i
WITH people, i,
     people[toInteger(rand() * size(people))] AS a
WITH people, a,
     CASE WHEN rand() < 0.85
       THEN [x IN people WHERE x.community = a.community AND x <> a]
       ELSE [x IN people WHERE x <> a]
     END AS candidates
WITH a, candidates[toInteger(rand() * size(candidates))] AS b
WHERE b IS NOT NULL AND a <> b
MERGE (a)-[:KNOWS]->(b);

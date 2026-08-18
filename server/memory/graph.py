"""NetworkX inter-visitor relationship graph."""

import networkx as nx
import pickle
from pathlib import Path
from datetime import datetime

class RelationshipGraph:
    def __init__(self, path: Path):
        self.path = path
        self.G = nx.DiGraph()
        self.G.add_node("patient", type="patient")

    def add_person(self, person_id: str, name: str, relation_to_patient: str, closeness: float = 0.5):
        self.G.add_node(
            person_id,
            name=name,
            type="person",
            closeness=closeness,
            added_at=datetime.now().isoformat()
        )

        self.G.add_edge(
            person_id, "patient",
            relation=relation_to_patient,
            edge_type="person_to_patient",
            confidence=1.0
        )


    def add_inter_person_edge(self, from_id: str, to_id: str, relation: str, confidence: float = 0.8, source: str = "extracted"):
        if not self.G.has_node(from_id) or not self.G.has_node(to_id):
            return
        
        if self.G.has_edge(from_id, to_id):
            self.G[from_id][to_id]["confidence"] = max(self.G[from_id][to_id]["confidence"], confidence)
            return
        
        self.G.add_edge(
            from_id, to_id,
            relation=relation,
            edge_type="person_to_person",
            confidence=confidence,
            source=source,
            extracted_at=datetime.now().isoformat()
        )

    def get_relation_to_patient(self, person_id: str) -> str:
        edge = self.G.get_edge_data(person_id, "patient")
        return edge["relation"] if edge else "unknown"


    def get_inter_person_relations(self, person_id: str) -> list[dict]:
        relations = []
        for neighbour in self.G.successors(person_id):
            if neighbour == "patient":
                continue
            edge = self.G[person_id][neighbour]
            relations.append({
                "person_id": neighbour,
                "name": self.G.nodes[neighbour].get("name", "unknown"),
                "relation": edge["relation"],
                "confidence": edge["confidence"]
            })
        return relations

    def narrow_unknown_face(self, co_present_person_ids: list[str]) -> list[str]:
        candidates = set()
        for known_id in co_present_person_ids:
            for neighbour in self.G.successors(known_id):
                if neighbour != "patient":
                    candidates.add(neighbour)
            for neighbour in list(candidates):
                for second_hop in self.G.successors(neighbour):
                    if second_hop != "patient":
                        candidates.add(second_hop)
        candidates -= set(co_present_person_ids)
        return list(candidates)
    
    def get_shared_connections(self, person_a: str, person_b: str) -> list[dict]:
        a_neighbor = set(self.G.successors(person_a)) - {"patient"}
        b_neighbor = set(self.G.successors(person_b)) - {"patient"}
        shared = a_neighbor & b_neighbor

        return [
            {"person_id": pid, "name": self.G.nodes[pid].get("name")}
            for pid in shared
        ]


    def get_social_context_for_recall(self, person_id: str) -> dict:
        relations = self.get_inter_person_relations(person_id)
        also_knows_patient = [
            r for r in relations if self.G.has_edge(r["person_id"], "patient")
        ]

        return {
            "relation_to_patient": self.get_relation_to_patient(person_id),
            "knows_others_who_visit": [
                {
                    "name": r["name"],
                    "how_connected": r["relation"],
                    "their_relation_to_patient": self.get_relation_to_patient(r["person_id"])
                }
                for r in also_knows_patient
            ],
            "other connections": [
                r for r in relations
                if not self.G.has_edge(r["person_id"], "patient")
            ]
        }

    def most_frequent_visitors(self, top_n: int = 3) -> list[str]:
        people = [n for n, d in self.G.nodes(data=True)
                    if d.get("type") == "person"]
        return people[:top_n]

    def set_patient_name(self, name: str):
        """Set the patient's name on the root node."""
        self.G.nodes["patient"]["name"] = name

    def get_patient_name(self) -> str:
        """Get the patient's name from the root node."""
        return self.G.nodes.get("patient", {}).get("name", "")

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.G, f)

    def load(self):
        with open(self.path, "rb") as f:
            self.G = pickle.load(f)


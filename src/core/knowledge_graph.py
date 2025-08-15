# FILE: src/core/knowledge_graph.py
import networkx as nx
from typing import List, Dict, Optional

class EcologicalKnowledgeGraph:
    """
    A knowledge graph to store and query complex ecological relationships,
    including species interactions, succession patterns, and symbiotic networks.
    """
    def __init__(self):
        """Initializes the knowledge graph."""
        self.graph = nx.DiGraph()
        self._populate_graph()
        print("✅ Ecological Knowledge Graph initialized with sample data.")

    def _populate_graph(self):
        """
        Populates the graph with sample ecological data. In a production system,
        this would be loaded from a comprehensive graph database.
        """
        # --- Node Types: species, interaction, ecosystem, invasive ---
        
        # Add species nodes
        self.graph.add_node("Shorea robusta", type='species', common_name='Sal')
        self.graph.add_node("Tectona grandis", type='species', common_name='Teak')
        self.graph.add_node("Vachellia nilotica", type='species', common_name='Babul', role='nitrogen-fixer')
        self.graph.add_node("Lantana camara", type='invasive', common_name='Lantana')
        
        # Add ecosystem nodes
        self.graph.add_node("Dry Deciduous Forest", type='ecosystem')
        
        # --- Edge Types: HAS_ROLE, INTERACTS_WITH, PIONEER_FOR, CLIMAX_IN ---
        
        # Succession Patterns
        self.graph.add_edge("Vachellia nilotica", "Dry Deciduous Forest", type='PIONEER_FOR')
        self.graph.add_edge("Shorea robusta", "Dry Deciduous Forest", type='CLIMAX_IN')
        
        # Symbiotic Networks (Nitrogen Fixing)
        self.graph.add_edge("Vachellia nilotica", "Shorea robusta", type='INTERACTS_WITH', relationship='facilitates_growth')
        
        # Invasive Species Filter
        # Lantana is known to inhibit the growth of Sal
        self.graph.add_edge("Lantana camara", "Shorea robusta", type='INTERACTS_WITH', relationship='inhibits_growth')

    def get_symbiotic_partners(self, species_name: str) -> List[str]:
        """Finds species that have a positive interaction (e.g., nitrogen-fixing)."""
        partners = []
        for u, v, data in self.graph.edges(data=True):
            if v == species_name and data.get('relationship') == 'facilitates_growth':
                partners.append(u)
        return partners

    def get_pioneer_species(self, ecosystem_name: str) -> List[str]:
        """Finds pioneer species for a given ecosystem."""
        pioneers = []
        for u, v, data in self.graph.edges(data=True):
            if v == ecosystem_name and data.get('type') == 'PIONEER_FOR':
                pioneers.append(u)
        return pioneers

    def check_invasive_interactions(self, species_list: List[str]) -> Dict[str, List[str]]:
        """Checks for negative interactions from known invasive species."""
        warnings = {}
        invasive_species = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'invasive']
        
        for invasive in invasive_species:
            for u, v, data in self.graph.edges(data=True):
                if u == invasive and v in species_list and data.get('relationship') == 'inhibits_growth':
                    if v not in warnings:
                        warnings[v] = []
                    warnings[v].append(f"Growth may be inhibited by the invasive species '{invasive}'.")
        return warnings

if __name__ == '__main__':
    # --- Smoke Test for the Ecological Knowledge Graph ---
    print("--- Initializing Knowledge Graph ---")
    kg = EcologicalKnowledgeGraph()
    
    # Test queries
    target_species = "Shorea robusta"
    target_ecosystem = "Dry Deciduous Forest"
    
    print(f"\n--- Querying for ecosystem: {target_ecosystem} ---")
    
    pioneers = kg.get_pioneer_species(target_ecosystem)
    print(f"  - Pioneer Species: {pioneers}")
    assert "Vachellia nilotica" in pioneers
    
    symbiotic_partners = kg.get_symbiotic_partners(target_species)
    print(f"  - Symbiotic Partners for {target_species}: {symbiotic_partners}")
    assert "Vachellia nilotica" in symbiotic_partners
    
    recommendation_list = ["Shorea robusta", "Tectona grandis"]
    invasive_warnings = kg.check_invasive_interactions(recommendation_list)
    print(f"  - Invasive Species Warnings for {recommendation_list}: {invasive_warnings}")
    assert "Shorea robusta" in invasive_warnings
    
    print("\n✅ Smoke test passed successfully.")

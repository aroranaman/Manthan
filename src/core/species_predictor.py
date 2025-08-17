# FILE: src/core/knowledge_recommender.py

from pathlib import Path
from .knowledge_graph import KnowledgeGraph # Import our new class

class KnowledgeRecommender:
    """
    Uses the KnowledgeGraph to generate intelligent, context-aware recommendations.
    """
    def __init__(self, data_dir: Path):
        self.kg = KnowledgeGraph(data_dir=data_dir)

    def recommend(self, site_features: dict, top_k: int = 10) -> dict:
        """
        Generates recommendations by querying the Knowledge Graph and ranking the results.
        """
        # 1. Get ecologically suitable candidates from the Knowledge Graph
        suitable_species = self.kg.apply_ecological_rules(site_features)

        if suitable_species.empty:
            return {"recommendations": [], "explanations": ["No suitable species found after applying ecological filters."]}

        # 2. Rank the candidates (e.g., prioritize threatened species)
        suitable_species['rank_score'] = suitable_species['is_threatened'].apply(lambda x: 1.0 if x else 0.5)
        
        final_recommendations = suitable_species.sort_values('rank_score', ascending=False).head(top_k)

        return {
            "recommendations": final_recommendations.to_dict('records'),
            "explanations": [f"Found {len(suitable_species)} ecologically suitable species.", f"Prioritizing {int(suitable_species['is_threatened'].sum())} threatened species."]
        }
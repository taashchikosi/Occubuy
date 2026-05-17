import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

class PropertyMatchingEngine:
    def __init__(self):
        self.property_kb = pd.read_csv(DATA_DIR / "occubuy_property_knowledge_base.csv")

    def match_properties_by_lifestyle(self, lifestyle_keywords: str, budget_range: Tuple[float, float] = None,
                                     top_k: int = 5) -> List[Dict]:
        """Match properties based on lifestyle preferences and optional budget"""

        keywords = [k.lower().strip() for k in lifestyle_keywords.split(',')]
        scores = self._calculate_lifestyle_scores(keywords)

        if budget_range:
            min_price, max_price = budget_range
            filtered = self.property_kb[
                (self.property_kb['price'] >= min_price) &
                (self.property_kb['price'] <= max_price)
            ].copy()
        else:
            filtered = self.property_kb.copy()

        filtered['lifestyle_match_score'] = filtered.index.map(lambda i: scores.get(i, 0))
        filtered = filtered.sort_values('lifestyle_match_score', ascending=False)

        results = []
        for _, row in filtered.head(top_k).iterrows():
            match_dict = row.to_dict()
            match_dict['match_reasoning'] = self._explain_match(row, keywords)
            results.append(match_dict)

        return results

    def _calculate_lifestyle_scores(self, keywords: List[str]) -> Dict[int, float]:
        """Calculate lifestyle match scores for all properties"""
        scores = {}

        for idx, row in self.property_kb.iterrows():
            score = 0
            archetype = str(row.get('ideal_buyer_archetype', '')).lower()
            lifestyle_supported = str(row.get('lifestyle_supported', '')).lower()
            desired_feeling = str(row.get('desired_home_feeling', '')).lower()
            match_summary = str(row.get('match_summary', '')).lower()

            text_to_search = f"{archetype} {lifestyle_supported} {desired_feeling} {match_summary}"

            for keyword in keywords:
                if keyword in text_to_search:
                    score += 1
                if keyword in archetype:
                    score += 2

            scores[idx] = score

        return scores

    def _explain_match(self, property_row: pd.Series, keywords: List[str]) -> str:
        """Generate human-readable explanation of the match"""
        archetype = property_row.get('ideal_buyer_archetype', 'Unknown')
        location = property_row.get('suburb', 'Unknown')
        price = property_row.get('price', 0)
        yield_pct = property_row.get('estimated_rental_yield_percent', 0)

        return (
            f"This {archetype} property in {location} (${price:,.0f}) aligns with your search. "
            f"Estimated rental yield: {yield_pct:.1f}%. "
            f"Perfect for: {property_row.get('lifestyle_supported', 'N/A')}"
        )

    def rank_properties_by_investment_quality(self, properties: List[Dict] = None,
                                            top_k: int = 10) -> List[Dict]:
        """Rank all or given properties by investment quality metrics"""

        if properties:
            property_ids = [p['property_id'] for p in properties]
            df = self.property_kb[self.property_kb['property_id'].isin(property_ids)].copy()
        else:
            df = self.property_kb.copy()

        df['investment_score'] = (
            df['capital_growth_score'] * 0.40 +
            df['estimated_rental_yield_percent'] * 10 * 0.30 +
            df['investment_quality_score'] * 0.30
        )

        df = df.sort_values('investment_score', ascending=False)

        results = []
        for _, row in df.head(top_k).iterrows():
            result = row.to_dict()
            result['investment_reasoning'] = (
                f"Growth score: {row['capital_growth_score']}/100, "
                f"Yield: {row['estimated_rental_yield_percent']:.2f}%, "
                f"Quality: {row['investment_quality_score']}/100"
            )
            results.append(result)

        return results

    def find_affordable_properties(self, budget: float, preferred_location: str = None,
                                  top_k: int = 10) -> List[Dict]:
        """Find properties within budget, optionally filtered by location"""

        filtered = self.property_kb[self.property_kb['price'] <= budget]

        if preferred_location:
            filtered = filtered[
                filtered['suburb'].str.contains(preferred_location, case=False, na=False)
            ]

        filtered = filtered.sort_values('investment_quality_score', ascending=False)

        results = []
        for _, row in filtered.head(top_k).iterrows():
            result = row.to_dict()
            result['affordability_note'] = (
                f"At ${row['price']:,.0f}, deposit needed: "
                f"10% (${row['estimated_deposit_10_percent']:,.0f}) or "
                f"20% (${row['estimated_deposit_20_percent']:,.0f})"
            )
            results.append(result)

        return results

    def get_property_recommendations(self, user_profile: Dict, financial_profile: Dict,
                                    top_k: int = 5) -> List[Dict]:
        """Generate personalized property recommendations based on user & financial profile"""

        max_budget = financial_profile['monthly_income'] * 5 if 'monthly_income' in financial_profile else 500000

        filtered = self.property_kb[self.property_kb['price'] <= max_budget].copy()

        filtered['recommendation_score'] = 0

        if 'lifestyle_supported' in user_profile:
            lifestyle_keywords = user_profile['lifestyle_supported'].lower()
            for idx, row in filtered.iterrows():
                supported = str(row.get('lifestyle_supported', '')).lower()
                if lifestyle_keywords in supported or any(k in supported for k in lifestyle_keywords.split(',')):
                    filtered.loc[idx, 'recommendation_score'] += 3

        if 'target_archetype' in user_profile:
            target_archetype = user_profile['target_archetype'].lower()
            for idx, row in filtered.iterrows():
                archetype = str(row.get('ideal_buyer_archetype', '')).lower()
                if target_archetype in archetype:
                    filtered.loc[idx, 'recommendation_score'] += 2

        filtered['recommendation_score'] += filtered['investment_quality_score'] / 25

        filtered = filtered.sort_values('recommendation_score', ascending=False)

        results = []
        for _, row in filtered.head(top_k).iterrows():
            result = row.to_dict()
            result['why_recommended'] = (
                f"Matches your profile: {row['ideal_buyer_archetype']}. "
                f"Strong investment fundamentals ({row['investment_quality_score']}/100). "
                f"Location: {row['suburb']}, {row['state']}."
            )
            results.append(result)

        return results

    def get_similar_properties(self, property_id: int, top_k: int = 5) -> List[Dict]:
        """Find properties similar to a given property"""

        target = self.property_kb[self.property_kb['property_id'] == property_id]
        if target.empty:
            return []

        target = target.iloc[0]

        # Match by archetype and price range (+/- 20%)
        price_min = target['price'] * 0.80
        price_max = target['price'] * 1.20
        archetype = target['ideal_buyer_archetype']

        similar = self.property_kb[
            (self.property_kb['price'] >= price_min) &
            (self.property_kb['price'] <= price_max) &
            (self.property_kb['ideal_buyer_archetype'] == archetype) &
            (self.property_kb['property_id'] != property_id)
        ].copy()

        similar = similar.sort_values('investment_quality_score', ascending=False)

        results = []
        for _, row in similar.head(top_k).iterrows():
            result = row.to_dict()
            price_diff = row['price'] - target['price']
            result['similarity_note'] = (
                f"Similar archetype & price. "
                f"Price difference: ${price_diff:+,.0f}. "
                f"Investment quality: {row['investment_quality_score']}/100"
            )
            results.append(result)

        return results

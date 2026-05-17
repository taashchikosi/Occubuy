import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

DATA_DIR = Path(__file__).parent / "data"

class FinancialAnalysisEngine:
    def __init__(self):
        self.financial_kb = pd.read_csv(DATA_DIR / "occubuy_financial_knowledge_base.csv")
        self.transactions = pd.read_csv(DATA_DIR / "occubuy_user_transactions.csv")
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        self.user_profiles = {row['user_id']: row.to_dict() for _, row in self.financial_kb.iterrows()}

    def get_user_financial_profile(self, user_id: int) -> Dict:
        if user_id not in self.user_profiles:
            return None
        profile = self.user_profiles[user_id].copy()
        user_txns = self.transactions[self.transactions['user_id'] == user_id]
        profile['last_12_months_transactions'] = len(user_txns)
        profile['recent_transactions'] = user_txns.tail(5).to_dict('records')
        return profile

    def calculate_affordability(self, user_id: int, property_price: float) -> Dict:
        profile = self.get_user_financial_profile(user_id)
        if not profile:
            return {"error": "User not found"}

        monthly_income = profile['average_monthly_income']
        monthly_surplus = profile['average_monthly_surplus']
        savings_rate = profile['savings_rate']

        deposit_10pct = property_price * 0.10
        deposit_20pct = property_price * 0.20

        estimated_annual_repayment = (property_price - deposit_20pct) * 0.06
        monthly_repayment_estimate = estimated_annual_repayment / 12
        repayment_to_income = monthly_repayment_estimate / monthly_income if monthly_income > 0 else 0

        months_to_10pct = deposit_10pct / (monthly_surplus * savings_rate) if monthly_surplus * savings_rate > 0 else float('inf')
        months_to_20pct = deposit_20pct / (monthly_surplus * savings_rate) if monthly_surplus * savings_rate > 0 else float('inf')

        readiness = self._determine_readiness(
            monthly_surplus, repayment_to_income, months_to_20pct,
            profile['estimated_emergency_buffer_months']
        )

        return {
            "property_price": property_price,
            "deposit_10pct": deposit_10pct,
            "deposit_20pct": deposit_20pct,
            "monthly_income": monthly_income,
            "monthly_surplus": monthly_surplus,
            "estimated_monthly_repayment": monthly_repayment_estimate,
            "repayment_to_income_ratio": repayment_to_income,
            "months_to_10pct_deposit": max(0, months_to_10pct),
            "months_to_20pct_deposit": max(0, months_to_20pct),
            "readiness_status": readiness,
            "financial_archetype": profile['financial_archetype'],
            "savings_rate": savings_rate,
            "debt_to_income": profile['debt_to_income_ratio'],
            "emergency_buffer_months": profile['estimated_emergency_buffer_months'],
        }

    def _determine_readiness(self, monthly_surplus: float, repayment_ratio: float,
                             months_to_deposit: float, emergency_buffer: float) -> str:
        if months_to_deposit <= 3 and repayment_ratio <= 0.30 and emergency_buffer >= 3:
            return "Ready"
        elif months_to_deposit <= 12 and repayment_ratio <= 0.35 and emergency_buffer >= 2:
            return "Emerging"
        else:
            return "Not Yet"

    def generate_financial_roadmap(self, user_id: int, property_price: float,
                                  target_deposit_pct: float = 0.20) -> Dict:
        profile = self.get_user_financial_profile(user_id)
        affordability = self.calculate_affordability(user_id, property_price)

        if "error" in affordability:
            return affordability

        target_deposit = property_price * target_deposit_pct
        current_savings = profile.get('total_savings_transfers', 0)
        deposit_gap = max(0, target_deposit - current_savings)

        monthly_savings = profile['average_monthly_savings']
        months_needed = deposit_gap / monthly_savings if monthly_savings > 0 else float('inf')

        monthly_budget_options = self._generate_budget_scenarios(
            affordability, deposit_gap, monthly_savings
        )

        roadmap_steps = [
            f"Current deposit savings: ${current_savings:,.0f}",
            f"Target deposit ({target_deposit_pct*100:.0f}%): ${target_deposit:,.0f}",
            f"Deposit gap: ${deposit_gap:,.0f}",
            f"Current monthly savings rate: ${monthly_savings:,.0f}",
            f"At current pace: {months_needed:.0f} months to reach target",
            f"Monthly mortgage estimate: ${affordability['estimated_monthly_repayment']:,.0f}",
            f"Your monthly surplus: ${affordability['monthly_surplus']:,.0f}",
            f"Readiness status: {affordability['readiness_status']}",
        ]

        next_steps = self._generate_next_steps(affordability['readiness_status'], months_needed)

        return {
            "property_price": property_price,
            "target_deposit": target_deposit,
            "deposit_gap": deposit_gap,
            "months_to_target": months_needed,
            "readiness_status": affordability['readiness_status'],
            "roadmap_steps": roadmap_steps,
            "next_steps": next_steps,
            "budget_scenarios": monthly_budget_options,
            "financial_profile_summary": profile['plain_english_financial_summary'] if 'plain_english_financial_summary' in profile else "",
        }

    def _generate_budget_scenarios(self, affordability: Dict, deposit_gap: float,
                                  current_monthly_savings: float) -> List[Dict]:
        scenarios = [
            {
                "name": "Current Pace",
                "monthly_savings_increase": 0,
                "months_to_deposit": max(0, deposit_gap / current_monthly_savings) if current_monthly_savings > 0 else float('inf'),
                "feasibility": "Realistic - no changes needed"
            },
            {
                "name": "Boost by 25%",
                "monthly_savings_increase": current_monthly_savings * 0.25,
                "months_to_deposit": max(0, deposit_gap / (current_monthly_savings * 1.25)) if current_monthly_savings > 0 else float('inf'),
                "feasibility": "Review dining/entertainment budget"
            },
            {
                "name": "Boost by 50%",
                "monthly_savings_increase": current_monthly_savings * 0.50,
                "months_to_deposit": max(0, deposit_gap / (current_monthly_savings * 1.50)) if current_monthly_savings > 0 else float('inf'),
                "feasibility": "Requires significant lifestyle adjustment"
            },
        ]
        return scenarios

    def _generate_next_steps(self, readiness_status: str, months_needed: float) -> List[str]:
        if readiness_status == "Ready":
            return [
                "1. Get pre-mortgage approval (locks in rates, shows sellers you're serious)",
                "2. Arrange building/pest inspections",
                "3. Make an offer",
                "4. Complete formal mortgage application"
            ]
        elif readiness_status == "Emerging":
            months_buffer = max(3, months_needed - 3)
            return [
                f"1. Target pre-approval date: {months_buffer:.0f} months from now",
                "2. Maintain/increase monthly savings discipline",
                "3. Review budget: can you redirect $500-1000/month to deposits?",
                f"4. In {months_buffer:.0f} months: get pre-approval, then start property hunting",
                "5. Build credit score: pay bills on time, reduce existing debts"
            ]
        else:
            return [
                "1. Explore properties in a more affordable price range first",
                "2. Increase monthly savings: review discretionary spending",
                "3. Set a 12-month savings sprint goal",
                "4. After 12 months: reassess affordability at higher price points",
                "5. Consider: can a partner's income help? Additional investment income?"
            ]

    def get_all_users(self) -> List[int]:
        return sorted(self.user_profiles.keys())

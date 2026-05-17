from typing import Dict, List
from financial_analysis_engine import FinancialAnalysisEngine

class RoadmapGenerator:
    def __init__(self):
        self.financial_engine = FinancialAnalysisEngine()

    def generate_dream_home_conversation(self, user_id: int, property_data: Dict) -> Dict:
        """Generate Layer 1: Dream Home Discovery conversation"""

        property_name = f"{property_data.get('ideal_buyer_archetype', 'Property')} in {property_data.get('suburb', 'Unknown')}"
        price = property_data.get('price', 0)

        conversation = {
            "phase": "Layer 1: Dream Home Discovery",
            "property": property_name,
            "price": price,
            "opening": (
                f"I found a great match for you: a **{property_data.get('ideal_buyer_archetype', 'property')}** "
                f"in {property_data.get('suburb', 'Unknown')}, listed at ${price:,.0f}.\n\n"
                f"**Why this matches you:**\n"
                f"- Lifestyle: {property_data.get('lifestyle_supported', 'N/A')}\n"
                f"- Emotional feel: {property_data.get('desired_home_feeling', 'N/A')}\n"
                f"- Best for: {property_data.get('match_summary', 'N/A')}\n\n"
                f"Ready to see if you can afford it? Let's move to Layer 2 →"
            ),
            "key_details": {
                "bedrooms": property_data.get('bedrooms'),
                "bathrooms": property_data.get('bathrooms'),
                "parking": property_data.get('parking'),
                "land_size_sqm": property_data.get('land_size_sqm'),
                "internal_size_sqm": property_data.get('internal_size_sqm'),
                "days_on_market": property_data.get('days_on_market'),
                "estimated_rental_yield": property_data.get('estimated_rental_yield_percent'),
                "growth_score": property_data.get('capital_growth_score'),
            }
        }

        return conversation

    def generate_financial_feasibility_conversation(self, user_id: int, property_price: float,
                                                   deposit_percentage: float = 0.20) -> Dict:
        """Generate Layer 2: Financial Feasibility & Roadmap conversation"""

        affordability = self.financial_engine.calculate_affordability(user_id, property_price)
        roadmap = self.financial_engine.generate_financial_roadmap(user_id, property_price, deposit_percentage)

        if "error" in affordability:
            return {"error": affordability["error"]}

        readiness = affordability['readiness_status']
        profile = self.financial_engine.get_user_financial_profile(user_id)

        conversation = {
            "phase": "Layer 2: Financial Feasibility & Roadmap",
            "readiness_status": readiness,
            "user_financial_snapshot": {
                "archetype": profile['financial_archetype'],
                "monthly_income": affordability['monthly_income'],
                "monthly_surplus": affordability['monthly_surplus'],
                "savings_rate": affordability['savings_rate'],
                "debt_to_income": affordability['debt_to_income'],
            },
            "property_affordability": {
                "price": property_price,
                "deposit_20_percent": affordability['deposit_20pct'],
                "monthly_repayment_estimate": affordability['estimated_monthly_repayment'],
                "repayment_to_income_ratio": affordability['repayment_to_income_ratio'],
                "months_to_deposit_20pct": affordability['months_to_20pct_deposit'],
            },
        }

        # Generate readiness-specific messaging
        if readiness == "Ready":
            conversation["summary"] = (
                f"✅ **You're Ready!**\n\n"
                f"Your financial position is strong for this property:\n"
                f"- Monthly surplus: ${affordability['monthly_surplus']:,.0f}\n"
                f"- Estimated monthly repayment: ${affordability['estimated_monthly_repayment']:,.0f} "
                f"({affordability['repayment_to_income_ratio']*100:.1f}% of income)\n"
                f"- Deposit status: You could move toward deposit savings immediately\n\n"
                f"**Next Steps:**\n"
                + "\n".join(roadmap['next_steps'][:4])
            )
        elif readiness == "Emerging":
            months_away = affordability['months_to_20pct_deposit']
            conversation["summary"] = (
                f"⏳ **You're Emerging**\n\n"
                f"You're on track! At your current savings pace:\n"
                f"- Target deposit (20%): ${affordability['deposit_20pct']:,.0f}\n"
                f"- Months to reach: {months_away:.0f} months\n"
                f"- Monthly surplus: ${affordability['monthly_surplus']:,.0f}\n"
                f"- Estimated repayment: ${affordability['estimated_monthly_repayment']:,.0f}/month\n\n"
                f"**Next Steps:**\n"
                + "\n".join(roadmap['next_steps'][:4])
            )
        else:  # Not Yet
            conversation["summary"] = (
                f"⏸️ **Not Yet (Let's Adjust)**\n\n"
                f"This property is a bit of a stretch right now:\n"
                f"- Monthly repayment estimate: ${affordability['estimated_monthly_repayment']:,.0f}\n"
                f"- Your monthly surplus: ${affordability['monthly_surplus']:,.0f}\n"
                f"- Months to 20% deposit: {affordability['months_to_20pct_deposit']:.0f} months\n\n"
                f"**Next Steps:**\n"
                + "\n".join(roadmap['next_steps'][:4])
            )

        conversation["budget_scenarios"] = roadmap['budget_scenarios']
        conversation["roadmap_steps"] = roadmap['roadmap_steps']

        return conversation

    def generate_layer_3_action_plan(self, user_id: int, property_price: float) -> Dict:
        """Generate Layer 3: Concrete Action Plan"""

        roadmap = self.financial_engine.generate_financial_roadmap(user_id, property_price)
        affordability = self.financial_engine.calculate_affordability(user_id, property_price)

        readiness = affordability['readiness_status']

        action_plan = {
            "phase": "Layer 3: Action Plan",
            "readiness_status": readiness,
        }

        if readiness == "Ready":
            action_plan["timeline"] = {
                "week_1_2": [
                    "Get pre-mortgage approval from 2-3 lenders",
                    "Lock in interest rate",
                    "Get property inspection booked"
                ],
                "week_3_4": [
                    "Complete building/pest inspections",
                    "Review inspection results with real estate agent",
                    "Make an offer (with financing conditional on approval)"
                ],
                "week_5_8": [
                    "Lender completes final approval",
                    "Complete settlement preparations",
                    "Exchange contracts"
                ],
                "week_9_12": [
                    "Settlement day",
                    "Property ownership transfer",
                    "Move in!"
                ]
            }
            action_plan["documents_needed"] = [
                "Proof of income (payslips, tax returns)",
                "Bank statements (last 3-6 months)",
                "Employment letter",
                "ID & proof of address",
                "Details of existing debts/liabilities"
            ]
        elif readiness == "Emerging":
            months_away = affordability['months_to_20pct_deposit']
            action_plan["timeline"] = {
                f"Month 1-{int(months_away)-3}": [
                    "Maintain current savings discipline",
                    "Review budget for extra $500-1000/month to allocate to deposits",
                    "Build/maintain credit score (pay bills on time)",
                    "Monitor property market trends"
                ],
                f"Month {int(months_away)-2}": [
                    "Get pre-mortgage approval from 2-3 lenders",
                    "Confirm deposit amount & savings progress",
                    "Review loan options (LVR 80-90%)"
                ],
                f"Month {int(months_away)-1}": [
                    "Get serious about property hunting",
                    "Attend open inspections",
                    "Finalize financing terms"
                ],
                f"Month {int(months_away)}+": [
                    "Make offers on properties",
                    "Complete due diligence",
                    "Proceed to settlement"
                ]
            }
            action_plan["documents_needed"] = [
                "Proof of income",
                "Bank statements (6+ months to show savings trend)",
                "Employment letter",
                "Credit report (get ahead of lenders)"
            ]
        else:  # Not Yet
            action_plan["timeline"] = {
                "Month 1-3": [
                    "Review budget: where can you cut $500-1000/month?",
                    "Set 12-month savings goal",
                    "Look for ways to increase income (side hustles, promotions)"
                ],
                "Month 4-6": [
                    "Build emergency fund to 3+ months of expenses",
                    "Pay down existing debts (especially credit cards)",
                    "Explore co-buying options (partner, family help?)"
                ],
                "Month 7-12": [
                    "Accelerate savings push",
                    "Research first-home buyer schemes in your state",
                    "Start pre-approval conversations with lenders"
                ],
                "Month 12+": [
                    "Reassess affordability at higher price points",
                    "Get formal pre-approval",
                    "Begin serious property hunting"
                ]
            }
            action_plan["documents_needed"] = [
                "Proof of income",
                "Bank statements (12 months to show savings trend)",
                "Budget forecast"
            ]

        return action_plan

    def generate_conversational_response(self, user_message: str, context: Dict,
                                        layer: int) -> str:
        """Generate conversational AI response based on layer and user message"""

        message_lower = user_message.lower()

        # Layer 1 responses
        if layer == 1:
            if any(word in message_lower for word in ['affordable', 'cost', 'price', 'can i']):
                return "Great question! Let's move to Layer 2 to analyze your finances against this property."
            elif any(word in message_lower for word in ['like', 'love', 'interested', 'yes']):
                return "Excellent! This property checks your boxes. Ready to see if it fits your budget? Layer 2 →"
            elif any(word in message_lower for word in ['something else', 'different', 'other']):
                return "Let me find more options that match your lifestyle preferences."
            else:
                return "Would you like to explore more properties, or learn about affordability for this one?"

        # Layer 2 responses
        elif layer == 2:
            readiness = context.get('readiness_status', 'Unknown')

            if any(word in message_lower for word in ['timeline', 'when', 'how long', 'months']):
                if readiness == "Ready":
                    return "You're ready now! You could move forward within weeks if you find the right property."
                elif readiness == "Emerging":
                    months = context.get('months_to_10pct_deposit', context.get('months_to_deposit', 18))
                    return f"At your current savings pace, you'll reach your 10% deposit in ~{months:.0f} months."
                else:
                    months = context.get('months_to_10pct_deposit', context.get('months_to_deposit', '24+'))
                    return f"This property is a stretch — the 10% deposit is ~{months:.0f} months away. Consider more affordable options or boost savings."

            elif any(word in message_lower for word in ['budget', 'cut', 'save', 'money', 'increase']):
                return "Smart thinking! Let me show you some budget scenarios and where to redirect funds."

            elif any(word in message_lower for word in ['next', 'what do i do', 'steps', 'action']):
                return "Let's move to Layer 3 to build your concrete action plan with timeline."

            elif any(word in message_lower for word in ['pre-approval', 'mortgage', 'approval']):
                return "Pre-approval is the next step after confirming your readiness. We'll cover that in Layer 3."

            else:
                return f"Your readiness status: **{readiness}**. Would you like to explore budget scenarios or move to the action plan?"

        return "I'm here to help. What would you like to know?"

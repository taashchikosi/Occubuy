import streamlit as st
import pandas as pd
from typing import Dict, List
import os

from financial_analysis_engine import FinancialAnalysisEngine
from rag_utils import RAGRetriever
from matching_engine import PropertyMatchingEngine
from roadmap_generator import RoadmapGenerator

st.set_page_config(
    page_title="Occubuy - Home Buying Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_engines():
    financial_engine = FinancialAnalysisEngine()
    rag_retriever = RAGRetriever()
    matching_engine = PropertyMatchingEngine()
    roadmap_gen = RoadmapGenerator()
    return financial_engine, rag_retriever, matching_engine, roadmap_gen

def initialize_session():
    if 'current_layer' not in st.session_state:
        st.session_state.current_layer = 1
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1

def render_layer1(financial_engine, rag_retriever, matching_engine, roadmap_gen):
    st.header("🏠 Layer 1: Dream Home Discovery")
    st.write("Tell me about your ideal home. What matters to you?")

    col1, col2 = st.columns([2, 1])

    with col1:
        lifestyle_query = st.text_input(
            "What's your ideal lifestyle?",
            placeholder="e.g., Family-friendly, peaceful, urban, creative, investment-focused"
        )

    with col2:
        budget = st.number_input(
            "Budget ($)",
            min_value=100000,
            max_value=5000000,
            value=1000000,
            step=50000
        )

    if st.button("Find My Dream Home", key="find_home"):
        if lifestyle_query:
            with st.spinner("Searching for your perfect match..."):
                matches = rag_retriever.retrieve_properties_by_lifestyle(
                    lifestyle_query,
                    top_k=5
                )

                if not matches:
                    matches = matching_engine.match_properties_by_lifestyle(
                        lifestyle_query,
                        budget_range=(0, budget),
                        top_k=5
                    )

                if matches:
                    st.success(f"Found {len(matches)} properties that match your preferences!")

                    for idx, prop in enumerate(matches, 1):
                        with st.expander(
                            f"#{idx} {prop.get('ideal_buyer_archetype', 'Property')} in {prop.get('suburb', 'Unknown')} - ${prop.get('price', 0):,.0f}",
                            expanded=(idx == 1)
                        ):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Property Details")
                                details = {
                                    "Bedrooms": prop.get('bedrooms'),
                                    "Bathrooms": prop.get('bathrooms'),
                                    "Parking": prop.get('parking'),
                                    "Internal Size": f"{prop.get('internal_size_sqm', 0):.0f} sqm",
                                    "Land Size": f"{prop.get('land_size_sqm', 0):.0f} sqm",
                                    "Year Built": prop.get('year_built'),
                                    "Days on Market": prop.get('days_on_market'),
                                }
                                for key, val in details.items():
                                    st.write(f"**{key}:** {val}")

                            with col2:
                                st.subheader("Investment & Lifestyle")
                                investment = {
                                    "Capital Growth": f"{prop.get('capital_growth_score', 0)}/100",
                                    "Rental Yield": f"{prop.get('estimated_rental_yield_percent', 0):.2f}%",
                                    "Investment Quality": f"{prop.get('investment_quality_score', 0)}/100",
                                    "Lifestyle Match": f"{prop.get('lifestyle_match_score', 0)}/100",
                                }
                                for key, val in investment.items():
                                    st.write(f"**{key}:** {val}")

                            st.write(f"**Why this matches you:** {prop.get('match_summary', 'N/A')}")
                            st.write(f"**Emotional feel:** {prop.get('desired_home_feeling', 'N/A')}")

                            if st.button(f"Explore Affordability for Property #{idx}", key=f"explore_{idx}"):
                                st.session_state.selected_property = prop
                                st.session_state.current_layer = 2
                                st.rerun()
                else:
                    st.warning("No properties found matching your criteria. Try different keywords!")
        else:
            st.info("Please describe your ideal lifestyle to get started!")

def render_layer2(financial_engine, rag_retriever, matching_engine, roadmap_gen):
    st.header("💰 Layer 2: Financial Feasibility & Roadmap")

    if not st.session_state.selected_property:
        st.warning("Please select a property from Layer 1 first.")
        if st.button("← Back to Layer 1"):
            st.session_state.current_layer = 1
            st.rerun()
        return

    property_data = st.session_state.selected_property
    property_name = f"{property_data.get('ideal_buyer_archetype', 'Property')} in {property_data.get('suburb', 'Unknown')}"
    property_price = property_data.get('price', 0)

    st.subheader(f"📍 {property_name}")
    st.write(f"**Listed at: ${property_price:,.0f}**")

    affordability = financial_engine.calculate_affordability(st.session_state.user_id, property_price)
    roadmap = financial_engine.generate_financial_roadmap(st.session_state.user_id, property_price)

    if "error" not in affordability:
        readiness = affordability['readiness_status']

        col1, col2, col3 = st.columns(3)

        with col1:
            if readiness == "Ready":
                st.success(f"### ✅ {readiness}")
            elif readiness == "Emerging":
                st.warning(f"### ⏳ {readiness}")
            else:
                st.error(f"### ⏸️ {readiness}")

        with col2:
            st.metric("Monthly Surplus", f"${affordability['monthly_surplus']:,.0f}")

        with col3:
            st.metric("Est. Monthly Repayment", f"${affordability['estimated_monthly_repayment']:,.0f}")

        st.subheader("Your Financial Position")
        tab1, tab2, tab3 = st.tabs(["Overview", "Deposit Timeline", "Budget Scenarios"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Income & Savings**")
                income_metrics = {
                    "Monthly Income": f"${affordability['monthly_income']:,.0f}",
                    "Monthly Surplus": f"${affordability['monthly_surplus']:,.0f}",
                    "Savings Rate": f"{affordability['savings_rate']*100:.1f}%",
                }
                for key, val in income_metrics.items():
                    st.write(f"- {key}: **{val}**")

            with col2:
                st.write("**Deposit Needs**")
                deposit_metrics = {
                    "20% Deposit Target": f"${affordability['deposit_20pct']:,.0f}",
                    "10% Deposit Target": f"${affordability['deposit_10pct']:,.0f}",
                    "Est. Months to 20%": f"{affordability['months_to_20pct_deposit']:.0f}",
                }
                for key, val in deposit_metrics.items():
                    st.write(f"- {key}: **{val}**")

        with tab2:
            st.write(f"**Timeline to {affordability.get('deposit_20pct', 0):,.0f} deposit:**")
            for step in roadmap['roadmap_steps']:
                st.write(f"- {step}")

        with tab3:
            st.write("**What if you saved more?**")
            for scenario in roadmap['budget_scenarios']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{scenario['name']}**")
                with col2:
                    months = scenario['months_to_deposit']
                    months_display = f"{months:.0f} months" if months != float('inf') else "∞"
                    st.write(f"{months_display}")
                with col3:
                    st.write(f"_{scenario['feasibility']}_")

        st.divider()
        st.subheader("What happens next?")

        if readiness == "Ready":
            st.success("""
            ### You're in a strong position! 🎉

            **Next Steps:**
            1. Get pre-mortgage approval from 2-3 lenders (locks rates, shows you're serious)
            2. Arrange building/pest inspections
            3. Make an offer
            4. Complete formal mortgage application
            """)

        elif readiness == "Emerging":
            months_away = affordability['months_to_20pct_deposit']
            st.warning(f"""
            ### You're on track! ⏳

            **Target timeline:** {months_away:.0f} months

            **To accelerate:**
            - Boost savings by $500-1000/month (review dining/entertainment)
            - Redirect investment funds toward deposit
            - Explore co-ownership options

            **Next Steps:**
            1. Get pre-approval 3 months before your target date
            2. Focus on savings discipline
            3. Monitor market for properties in this range
            """)

        else:
            st.error("""
            ### Let's reframe this 📊

            This property is a stretch at current savings. But you have options!

            **Next Steps:**
            1. Explore properties in a more affordable range
            2. Review budget: can you redirect $500-1000/month?
            3. Consider 12-month savings sprint
            4. Explore co-buying with partner/family
            """)

        st.divider()
        st.subheader("Ask me anything about affordability")

        user_input = st.text_input(
            "Your question:",
            placeholder="e.g., Can I afford this? What if I save more? Show me cheaper options...",
            key="layer2_input"
        )

        if user_input:
            response = roadmap_gen.generate_conversational_response(user_input, affordability, layer=2)
            st.info(f"**Assistant:** {response}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Layer 1"):
                st.session_state.current_layer = 1
                st.rerun()

        with col2:
            if st.button("Proceed to Layer 3 (Action Plan) →"):
                st.session_state.current_layer = 3
                st.rerun()

    else:
        st.error(f"Error: {affordability['error']}")

def render_layer3(financial_engine, rag_retriever, matching_engine, roadmap_gen):
    st.header("📋 Layer 3: Action Plan")

    if not st.session_state.selected_property:
        st.warning("Please select a property from Layer 1 first.")
        if st.button("← Back to Layer 1"):
            st.session_state.current_layer = 1
            st.rerun()
        return

    property_data = st.session_state.selected_property
    property_name = f"{property_data.get('ideal_buyer_archetype', 'Property')} in {property_data.get('suburb', 'Unknown')}"
    property_price = property_data.get('price', 0)

    st.subheader(f"📍 {property_name}")

    action_plan = roadmap_gen.generate_layer_3_action_plan(st.session_state.user_id, property_price)
    readiness = action_plan['readiness_status']

    st.subheader(f"Your Timeline ({readiness})")

    timeline = action_plan['timeline']
    for phase, actions in timeline.items():
        with st.expander(phase, expanded=True):
            for action in actions:
                st.write(f"✓ {action}")

    st.subheader("Documents You'll Need")
    for doc in action_plan['documents_needed']:
        st.write(f"- {doc}")

    st.subheader("Pre-Purchase Checklist")
    checklist_items = [
        "Get pre-mortgage approval",
        "Arrange property inspection",
        "Review inspection results",
        "Finalize financing terms",
        "Get home and contents insurance quotes",
        "Arrange pest/building report",
        "Exchange contracts",
        "Final settlement preparations",
        "Settlement day!",
    ]

    cols = st.columns(2)
    for idx, item in enumerate(checklist_items):
        with cols[idx % 2]:
            st.checkbox(item)

    st.divider()
    st.subheader("Your Immediate Next Actions")

    if readiness == "Ready":
        st.success("""
        1. **Today:** Research 2-3 mortgage lenders
        2. **This week:** Get pre-approval (have documents ready)
        3. **Next week:** Schedule property inspection
        4. **Within 2 weeks:** Make an offer
        """)
    elif readiness == "Emerging":
        st.warning("""
        1. **This month:** Review budget for savings optimization
        2. **Next month:** Start conversations with lenders (even if not ready yet)
        3. **In 2-3 months:** Get formal pre-approval
        4. **In months:** Begin property hunting
        """)
    else:
        st.error("""
        1. **This week:** Review your full budget
        2. **Next week:** Identify $500-1000/month to redirect to savings
        3. **This month:** Research first-home buyer schemes in your state
        4. **Month 3+:** Get pre-approval conversations started
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Layer 2"):
            st.session_state.current_layer = 2
            st.rerun()

    with col3:
        if st.button("← Start Over (Layer 1)"):
            st.session_state.selected_property = None
            st.session_state.current_layer = 1
            st.rerun()

def render_debug_tab(financial_engine, rag_retriever, matching_engine, roadmap_gen):
    st.header("🔧 Debug & Data")

    tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Properties", "Transactions", "Logs"])

    with tab1:
        st.subheader("User Financial Profiles")
        users = financial_engine.get_all_users()
        selected_user = st.selectbox("Select user:", users)

        if selected_user:
            profile = financial_engine.get_user_financial_profile(selected_user)
            st.json({k: v for k, v in profile.items() if not isinstance(v, (list, dict))})

    with tab2:
        st.subheader("Property Knowledge Base")
        properties = rag_retriever.property_kb
        st.write(f"Total properties: {len(properties)}")
        st.dataframe(
            properties[['property_id', 'suburb', 'price', 'bedrooms', 'investment_quality_score']],
            use_container_width=True
        )

    with tab3:
        st.subheader("Transaction Data")
        transactions = rag_retriever.financial_kb
        st.write(f"Total financial profiles: {len(transactions)}")
        st.dataframe(transactions[['user_id', 'financial_archetype', 'annual_income', 'savings_rate']], use_container_width=True)

    with tab4:
        st.subheader("System Status")
        st.write("✅ Financial Analysis Engine: Ready")
        st.write("✅ RAG Retriever: Ready")
        st.write("✅ Property Matching: Ready")
        st.write("✅ Roadmap Generator: Ready")
        if rag_retriever.faiss_index:
            st.write("✅ Vector DB (FAISS): Loaded")
        else:
            st.write("⚠️ Vector DB: Using TF-IDF fallback")

def main():
    initialize_session()

    financial_engine, rag_retriever, matching_engine, roadmap_gen = load_engines()

    st.sidebar.title("🏠 Occubuy")
    st.sidebar.write("Your AI home buying assistant")

    page = st.sidebar.radio(
        "Choose your path:",
        ["🏠 Layers 1-3 (Home Buying Flow)", "🔧 Debug & Data"]
    )

    if page == "🏠 Layers 1-3 (Home Buying Flow)":
        st.sidebar.markdown("---")
        st.sidebar.write(f"**Current Layer:** {st.session_state.current_layer}/3")
        progress = st.session_state.current_layer / 3
        st.sidebar.progress(progress)

        if st.session_state.current_layer == 1:
            render_layer1(financial_engine, rag_retriever, matching_engine, roadmap_gen)
        elif st.session_state.current_layer == 2:
            render_layer2(financial_engine, rag_retriever, matching_engine, roadmap_gen)
        else:
            render_layer3(financial_engine, rag_retriever, matching_engine, roadmap_gen)

    else:
        render_debug_tab(financial_engine, rag_retriever, matching_engine, roadmap_gen)

    st.sidebar.markdown("---")
    st.sidebar.write("Built with ❤️ for home buyers")

if __name__ == "__main__":
    main()

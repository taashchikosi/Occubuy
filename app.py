import streamlit as st
import os
from typing import Dict, List, Optional

from financial_analysis_engine import FinancialAnalysisEngine
from rag_utils import RAGRetriever
from matching_engine import PropertyMatchingEngine

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="Occubuy",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Engines ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_engines():
    return FinancialAnalysisEngine(), RAGRetriever(), PropertyMatchingEngine()


def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ── OpenAI calls ───────────────────────────────────────────────────────────────

def ai_chat(client, system_prompt: str, history: List[Dict], user_message: str) -> str:
    """Call OpenAI with conversation history. History is list of {role, text} dicts."""
    if not client:
        return "_OpenAI unavailable — please set OPENAI_API_KEY._"
    try:
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": h["role"], "content": h["text"]})
        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=600,
            temperature=0.75,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"_Error contacting OpenAI: {e}_"


def ai_extract_keywords(client, history: List[Dict]) -> str:
    """Ask the model to distil lifestyle search keywords from the Layer 1 conversation."""
    transcript = "\n".join(f"{h['role'].upper()}: {h['text']}" for h in history)
    prompt = (
        "From this home buyer conversation extract 6–10 comma-separated keywords that describe "
        "their ideal property lifestyle (e.g. family, peaceful, garden, suburban, near schools, "
        "urban, spacious, character home). Return ONLY the keywords, nothing else.\n\n"
        f"Conversation:\n{transcript}"
    )
    if not client:
        return " ".join(h["text"] for h in history if h["role"] == "user")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return " ".join(h["text"] for h in history if h["role"] == "user")


def ai_match_explanation(client, prop: Dict, history: List[Dict], rank: int) -> str:
    """Generate a personalised reason why this property suits this person."""
    transcript = "\n".join(f"{h['role'].upper()}: {h['text']}" for h in history[-8:])
    rank_word = ["best", "second", "third"][rank]
    prop_desc = (
        f"Property type: {prop.get('ideal_buyer_archetype', '')}\n"
        f"Suburb: {prop.get('suburb', '')}\n"
        f"Lifestyle supported: {prop.get('lifestyle_supported', '')}\n"
        f"Home feeling: {prop.get('desired_home_feeling', '')}\n"
        f"Summary: {prop.get('match_summary', '')}"
    )
    prompt = (
        f"Based on this conversation, write 1–2 sentences explaining why this is the {rank_word} "
        f"match for this person. Be specific to what they shared. Warm, conversational tone.\n\n"
        f"Conversation:\n{transcript}\n\nProperty:\n{prop_desc}\n\nExplanation only:"
    )
    if not client:
        return prop.get("match_summary", "")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return prop.get("match_summary", "")


# ── System prompts ─────────────────────────────────────────────────────────────

LAYER1_PROMPT = """You are Occubuy, a warm and empathetic home advisor. Your only job right now is to help this person emotionally and conversationally discover the kind of home and lifestyle that genuinely fits the life they want to build.

Through natural conversation, gently explore:
- Their current living situation and what they want to change
- Their daily life — how they actually use their home, morning routines, evenings, weekends
- Family situation: partner, children, pets, family nearby
- What "home" means to them emotionally — how they want to feel walking through the door
- Their ideal neighbourhood: quiet streets, urban energy, near nature, schools, cafes, walkability
- Their vision for the next 5–10 years
- What matters most: outdoor space, natural light, character, modern design, room to grow

Rules:
- Ask ONE question at a time
- Keep responses SHORT: 1–2 warm sentences, then your question
- Be genuinely curious — make them feel deeply heard, not like they're filling in a form
- Reference what they've said in your follow-up to show you're truly listening
- NEVER mention prices, budgets, mortgages, deposits, finance, or money in any form
- After 6–8 meaningful exchanges where you have a clear picture, end your message with the exact token: [MATCH_READY]"""


def layer2_prompt(prop: Dict, profile: Dict, affordability: Dict) -> str:
    price = prop.get("price", 0)
    name = f"{prop.get('ideal_buyer_archetype', 'Property')} in {prop.get('suburb', 'Unknown')}"
    readiness = affordability.get("readiness_status", "Unknown")
    m10 = affordability.get("months_to_10pct_deposit", 0)
    m20 = affordability.get("months_to_20pct_deposit", 0)
    monthly_income = affordability.get("monthly_income", 0)
    monthly_savings = affordability.get("monthly_savings", 0)
    current_savings = affordability.get("current_savings", 0)
    repayment = affordability.get("estimated_monthly_repayment", 0)
    repayment_ratio = affordability.get("repayment_to_income_ratio", 0)
    dti = profile.get("debt_to_income_ratio", 0)
    buffer = profile.get("estimated_emergency_buffer_months", 0)

    return f"""You are Occubuy, a knowledgeable and empathetic home buying advisor. The user has chosen their dream home and you are now helping them understand their financial readiness.

THEIR CHOSEN HOME:
- {name}
- Price: ${price:,.0f}
- Estimated monthly repayment: ${repayment:,.0f}/mo

THEIR FINANCIAL PROFILE (from their real data):
- Financial archetype: {profile.get('financial_archetype', 'Unknown')}
- Monthly income: ${monthly_income:,.0f}
- Monthly savings rate: ${monthly_savings:,.0f}/mo
- Current savings balance: ${current_savings:,.0f}
- 10% deposit target: ${affordability.get('deposit_10pct', 0):,.0f} — approx {m10:.0f} months away at current pace
- 20% deposit target: ${affordability.get('deposit_20pct', 0):,.0f} — approx {m20:.0f} months away
- Repayment as % of income: {repayment_ratio * 100:.1f}%
- Debt-to-income ratio: {dti * 100:.1f}%
- Emergency buffer: {buffer:.1f} months
- Readiness status: {readiness}

YOUR ROLE — conduct a genuine financial readiness conversation:
1. Ask about their income, savings, debts, job security, partner income, upcoming expenses — one question at a time
2. Use the real data above to give specific, accurate, personalised guidance
3. Build toward an honest picture of their readiness

IF READY: Congratulate them warmly. Walk them step by step through the purchase process — pre-approval, inspections, making an offer, exchange, settlement. Explain what pre-mortgage approval is and why to get it first.

IF EMERGING ({m10:.0f} months to 10% deposit): Be encouraging — they're on track. Explain the 10% deposit path (with LMI) vs the 20% path (no LMI). Give a real timeline. Suggest specific ways to accelerate savings.

IF NOT YET: Be compassionate and direct. Explain specifically what needs to change with real numbers. Give a realistic roadmap — what to save, over how long. Consider whether a more affordable property makes sense. Talk about government schemes, co-buying, income growth.

Rules:
- ONE question at a time — this is still a conversation, not a report
- Use real numbers from the financial data — be specific
- Stay empathetic — buying a home is emotional and stressful
- Reveal the picture gradually through conversation, don't dump everything at once"""


# ── Session state ──────────────────────────────────────────────────────────────

def init():
    defaults = {
        "messages": [],        # Chat display history [{role, type, content}]
        "stage": "layer1",     # layer1 | selecting | layer2
        "l1_history": [],      # AI Layer 1 exchange history
        "l2_history": [],      # AI Layer 2 exchange history
        "exchanges": 0,        # Layer 1 exchange count
        "properties": [],      # Top 3 matched properties
        "explanations": [],    # Personalised match explanations
        "selected": None,      # Chosen property
        "user_id": 1,
        "affordability": None,
        "profile": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_property_card(prop: Dict, explanation: str, rank: int):
    labels = ["⭐ Best Match", "✦ Second Pick", "◈ Third Pick"]
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{labels[rank]} — {prop.get('ideal_buyer_archetype', 'Property')}**")
            st.markdown(
                f"📍 {prop.get('suburb', 'Unknown')}, {prop.get('state', '')} &nbsp;|&nbsp; "
                f"💰 ${prop.get('price', 0):,.0f}"
            )
        with col2:
            st.markdown(
                f"🛏 {prop.get('bedrooms', '?')} &nbsp; 🚿 {prop.get('bathrooms', '?')}"
            )
        if explanation:
            st.caption(explanation)


def render_msg(msg: Dict):
    t = msg.get("type", "text")
    c = msg["content"]
    if t == "text":
        st.markdown(c)
    elif t == "properties":
        st.markdown(c["intro"])
        for i, (prop, exp) in enumerate(zip(c["props"], c["explanations"])):
            render_property_card(prop, exp, i)
        st.markdown(c["prompt"])


# ── Layer handlers ─────────────────────────────────────────────────────────────

def handle_layer1(user_input: str, client, rag_retriever, matching_engine) -> List[Dict]:
    st.session_state.exchanges += 1

    response = ai_chat(client, LAYER1_PROMPT, st.session_state.l1_history, user_input)

    st.session_state.l1_history.append({"role": "user", "text": user_input})
    st.session_state.l1_history.append({"role": "assistant", "text": response})

    ready = "[MATCH_READY]" in response or st.session_state.exchanges >= 9
    clean = response.replace("[MATCH_READY]", "").strip()

    out = [{"role": "assistant", "type": "text", "content": clean}]

    if ready:
        keywords = ai_extract_keywords(client, st.session_state.l1_history)

        matches = rag_retriever.retrieve_properties_by_lifestyle(keywords, top_k=5)
        if not matches:
            matches = matching_engine.match_properties_by_lifestyle(keywords, top_k=5)

        top3 = matches[:3]
        explanations = [
            ai_match_explanation(client, p, st.session_state.l1_history, i)
            for i, p in enumerate(top3)
        ]

        st.session_state.properties = top3
        st.session_state.explanations = explanations
        st.session_state.stage = "selecting"

        out.append({
            "role": "assistant",
            "type": "properties",
            "content": {
                "intro": "Based on everything you've shared, here are your top 3 matches:",
                "props": top3,
                "explanations": explanations,
                "prompt": "Which one feels most like home to you?",
            },
        })

    return out


def select_property(idx: int, client, financial_engine) -> List[Dict]:
    prop = st.session_state.properties[idx]
    st.session_state.selected = prop

    affordability = financial_engine.calculate_affordability(
        st.session_state.user_id, prop.get("price", 0)
    )
    profile = financial_engine.get_user_financial_profile(st.session_state.user_id)
    st.session_state.affordability = affordability
    st.session_state.profile = profile
    st.session_state.stage = "layer2"

    system = layer2_prompt(prop, profile, affordability)
    opening = ai_chat(
        client,
        system,
        [],
        (
            f"The user has just chosen their dream home: {prop.get('ideal_buyer_archetype', 'property')} "
            f"in {prop.get('suburb', 'Unknown')} at ${prop.get('price', 0):,.0f}. "
            "Acknowledge their exciting choice warmly (1–2 sentences), then ask your first question "
            "to begin understanding their financial situation. Keep it to 3–4 sentences total."
        ),
    )

    st.session_state.l2_history = [
        {"role": "user", "text": "[property selected]"},
        {"role": "assistant", "text": opening},
    ]

    return [
        {
            "role": "user",
            "type": "text",
            "content": f"I'd love to go with the {prop.get('ideal_buyer_archetype', 'property')} in {prop.get('suburb', 'Unknown')}.",
        },
        {"role": "assistant", "type": "text", "content": opening},
    ]


def handle_layer2(user_input: str, client, financial_engine) -> List[Dict]:
    system = layer2_prompt(
        st.session_state.selected,
        st.session_state.profile,
        st.session_state.affordability,
    )
    response = ai_chat(client, system, st.session_state.l2_history, user_input)

    st.session_state.l2_history.append({"role": "user", "text": user_input})
    st.session_state.l2_history.append({"role": "assistant", "text": response})

    return [{"role": "assistant", "type": "text", "content": response}]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init()
    financial_engine, rag_retriever, matching_engine = load_engines()
    client = get_openai_client()

    # Sidebar
    with st.sidebar:
        st.title("🏠 Occubuy")
        st.caption("AI home buying assistant")
        st.divider()

        users = financial_engine.get_all_users()
        uid = st.selectbox("Demo profile:", users, index=0)
        if uid != st.session_state.user_id:
            st.session_state.user_id = uid

        st.divider()
        stage_labels = {
            "layer1": "🏠 Layer 1: Discovering your home",
            "selecting": "🏠 Layer 1: Choose your home",
            "layer2": "💰 Layer 2: Financial readiness",
        }
        st.info(stage_labels.get(st.session_state.stage, ""))

        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if not client:
            st.error("⚠️ OPENAI_API_KEY not set")

    # Seed welcome message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": (
                "👋 Hi! I'm **Occubuy** — your home buying assistant.\n\n"
                "Before we look at anything, I want to get to know *you* a little. "
                "The best home isn't just about rooms and price — it's about the life you want to live.\n\n"
                "**So let's start simply: what does your life look like right now, "
                "and what are you hoping to change or create with a new home?**"
            ),
        })

    # Render all messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_msg(msg)

    # Property selection buttons — shown below the chat when in selecting stage
    if st.session_state.stage == "selecting" and st.session_state.properties:
        st.divider()
        st.markdown("**Choose the one that feels right:**")
        labels = ["⭐ Best Match", "✦ Second Pick", "◈ Third Pick"]
        cols = st.columns(3)
        for i, (prop, col) in enumerate(zip(st.session_state.properties, cols)):
            with col:
                if st.button(
                    f"{labels[i]}\n\n{prop.get('ideal_buyer_archetype', 'Property')}\n{prop.get('suburb', '')}",
                    key=f"pick_{i}",
                    use_container_width=True,
                ):
                    with st.spinner("Getting your financial picture ready..."):
                        new_msgs = select_property(i, client, financial_engine)
                    for m in new_msgs:
                        st.session_state.messages.append(m)
                    st.rerun()

    # Chat input — hidden only during property selection
    if st.session_state.stage != "selecting":
        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

            with st.spinner("..."):
                if st.session_state.stage == "layer1":
                    new_msgs = handle_layer1(prompt, client, rag_retriever, matching_engine)
                elif st.session_state.stage == "layer2":
                    new_msgs = handle_layer2(prompt, client, financial_engine)
                else:
                    new_msgs = []

            for m in new_msgs:
                st.session_state.messages.append(m)

            st.rerun()


if __name__ == "__main__":
    main()

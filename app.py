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
    initial_sidebar_state="expanded"
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


# ── AI calls ───────────────────────────────────────────────────────────────────

def ai_chat(client, system_prompt: str, history: List[Dict], user_message: str) -> str:
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
    transcript = "\n".join(f"{h['role'].upper()}: {h['text']}" for h in history)
    prompt = (
        "From this home buyer conversation extract 6–10 comma-separated keywords describing "
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

def layer1_prompt(name: str) -> str:
    first = name.split()[0]
    return f"""You are Occubuy, a warm and empathetic home advisor. You are speaking with {first}.

Your only job right now is to help {first} emotionally and conversationally discover the kind of home and lifestyle that genuinely fits the life they want to build.

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
- Use {first}'s name naturally but sparingly (once every few exchanges)
- Be genuinely curious — make them feel deeply heard, not like they're filling in a form
- Reference what they've said in your follow-up to show you're truly listening
- NEVER mention prices, budgets, mortgages, deposits, finance, or money in any form
- After 6–8 meaningful exchanges where you have a clear picture, end your message with the exact token: [MATCH_READY]"""


def layer2_prompt(prop: Dict, profile: Dict, affordability: Dict, scenarios: Dict, alternative: Dict = None) -> str:
    name = profile.get("name", "there")
    first = name.split()[0]
    price = prop.get("price", 0)
    prop_name = f"{prop.get('ideal_buyer_archetype', 'Property')} in {prop.get('suburb', 'Unknown')}"
    readiness = affordability.get("readiness_status", "Unknown")
    monthly_income = affordability.get("monthly_income", 0)
    current_savings = affordability.get("current_savings", 0)
    base_savings = affordability.get("monthly_savings", 0)
    repayment = affordability.get("estimated_monthly_repayment", 0)
    repayment_pct = affordability.get("repayment_to_income_ratio", 0) * 100
    dti = profile.get("debt_to_income_ratio", 0) * 100
    buffer = profile.get("estimated_emergency_buffer_months", 0)
    deposit_10 = affordability.get("deposit_10pct", 0)
    deposit_20 = affordability.get("deposit_20pct", 0)

    # Build scenario table from pre-computed accurate figures
    rows = []
    for s in scenarios.get("scenarios", []):
        m10 = f"{s['months_to_10pct']} months" if s["months_to_10pct"] else "N/A"
        m20 = f"{s['months_to_20pct']} months" if s["months_to_20pct"] else "N/A"
        rows.append(f"  {s['label']:<22} | ${s['monthly_savings']:>7,.0f}/mo | {m10:>12} | {m20:>12}")
    scenario_table = "\n".join(rows)

    return f"""You are Occubuy, a knowledgeable and empathetic home buying advisor speaking with {first}.

CHOSEN HOME:
- {prop_name}
- Price: ${price:,.0f}
- Estimated monthly repayment: ${repayment:,.0f}/mo ({repayment_pct:.1f}% of {first}'s income)

{first.upper()}'S VERIFIED FINANCIAL PROFILE:
- Monthly income: ${monthly_income:,.0f}
- Current savings balance: ${current_savings:,.0f}
- Monthly savings rate: ${base_savings:,.0f}/mo
- Debt-to-income ratio: {dti:.1f}%
- Emergency buffer: {buffer:.1f} months
- Financial archetype: {profile.get('financial_archetype', 'Unknown')}
- Readiness status: {readiness}

DEPOSIT SAVINGS SCENARIOS — pre-computed, mathematically accurate. Use ONLY these figures:

  Scenario               | Monthly Savings | To 10% deposit | To 20% deposit
  -----------------------|-----------------|----------------|----------------
{scenario_table}

  10% deposit = ${deposit_10:,.0f} → enter sooner with LMI (Lenders Mortgage Insurance)
  20% deposit = ${deposit_20:,.0f} → no LMI required, longer wait

YOUR APPROACH:
You already know {first}'s complete financial picture from their verified data. Do NOT ask questions you already have the answers to — don't ask about their income, savings rate, or how much they save. You know this.

Instead, have a real financial discussion:
1. Open by presenting their position clearly and warmly — reference their name and specific numbers
2. Explain what it means for this specific property and price point
3. Discuss their options using the pre-computed scenario table above
4. Respond to their questions, explore concerns, discuss trade-offs conversationally

BASED ON READINESS ({readiness}):

IF READY: Congratulate {first} genuinely. Walk through the purchase process step by step in a conversational way — what pre-mortgage approval is and why to get it first, building and pest inspections, making an offer, exchanging contracts, settlement. Make it feel exciting and achievable.

IF EMERGING: Present the deposit timeline honestly using the scenario table. Discuss the 10% LMI path vs waiting for 20% — help {first} decide which suits their life. Mention first home buyer schemes if applicable. Be encouraging — they're on track.

IF NOT YET: Be compassionate and direct. Show {first} the gap with the exact numbers. Discuss what levers they can pull — save more (show the scenarios), consider a lower price point, co-buying, government schemes, income growth. Give them a real roadmap they can act on.

{"" if readiness != "Not Yet" or not alternative else f"""
ALTERNATIVE PROPERTY WITHIN THEIR BUDGET:
Since {first} isn't ready for the dream home yet, you have a more affordable alternative to suggest:
- {alternative.get('ideal_buyer_archetype', 'Property')} in {alternative.get('suburb', 'Unknown')}, {alternative.get('state', '')}
- Price: ${alternative.get('price', 0):,.0f} (vs ${price:,.0f} for the dream home)
- Lifestyle: {alternative.get('lifestyle_supported', 'N/A')}
- Match summary: {alternative.get('match_summary', 'N/A')}

After presenting {first}'s financial position honestly, naturally suggest this as a smart stepping stone — not giving up on the dream, but a property they could own NOW that still fits their lifestyle. Frame it positively and practically.
"""}
Rules:
- ONE topic per response — guide the conversation, don't dump everything at once
- Use {first}'s name naturally (once every few exchanges, not every message)
- Use ONLY the pre-computed figures above — never invent or recalculate numbers yourself
- Stay empathetic throughout — this is one of the biggest decisions of their life"""


# ── Session state ──────────────────────────────────────────────────────────────

def init():
    defaults = {
        "messages": [],
        "stage": "layer1",
        "l1_history": [],
        "l2_history": [],
        "exchanges": 0,
        "properties": [],
        "explanations": [],
        "selected": None,
        "user_id": 1,
        "affordability": None,
        "profile": None,
        "scenarios": None,
        "alternative_property": None,
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
            st.markdown(f"🛏 {prop.get('bedrooms', '?')} &nbsp; 🚿 {prop.get('bathrooms', '?')}")
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

def handle_layer1(user_input: str, client, rag_retriever, matching_engine, name: str) -> List[Dict]:
    st.session_state.exchanges += 1
    response = ai_chat(client, layer1_prompt(name), st.session_state.l1_history, user_input)

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


def select_property(idx: int, client, financial_engine, matching_engine) -> List[Dict]:
    prop = st.session_state.properties[idx]
    st.session_state.selected = prop

    affordability = financial_engine.calculate_affordability(
        st.session_state.user_id, prop.get("price", 0)
    )
    profile = financial_engine.get_user_financial_profile(st.session_state.user_id)
    scenarios = financial_engine.compute_scenarios(
        st.session_state.user_id, prop.get("price", 0)
    )
    # Find affordable alternative if user is Not Yet ready
    alternative = None
    if affordability.get("readiness_status") == "Not Yet":
        dream_price = prop.get("price", 0)
        alt_ceiling = dream_price * 0.80  # at least 20% cheaper to be meaningful
        lifestyle_keywords = " ".join(
            h["text"] for h in st.session_state.l1_history if h["role"] == "user"
        )
        alt_props = matching_engine.match_properties_by_lifestyle(
            lifestyle_keywords, budget_range=(0, alt_ceiling), top_k=1
        )
        if not alt_props:
            # Fallback: cheapest available property that isn't the same one
            alt_props = matching_engine.find_affordable_properties(dream_price * 0.99, top_k=3)
            alt_props = [p for p in alt_props if p.get("price", 0) < dream_price][:1]
        if alt_props:
            alternative = alt_props[0]

    st.session_state.affordability = affordability
    st.session_state.profile = profile
    st.session_state.scenarios = scenarios
    st.session_state.alternative_property = alternative
    st.session_state.stage = "layer2"

    system = layer2_prompt(prop, profile, affordability, scenarios, alternative)
    first = profile.get("name", "there").split()[0]
    opening = ai_chat(
        client, system, [],
        (
            f"The user has just chosen their dream home: {prop.get('ideal_buyer_archetype', 'property')} "
            f"in {prop.get('suburb', 'Unknown')} at ${prop.get('price', 0):,.0f}. "
            f"Open by warmly acknowledging {first}'s choice (1 sentence), then immediately present "
            f"their financial position using their real numbers — tell them where they stand clearly "
            f"and empathetically. Based on their readiness status, set the direction of the conversation. "
            + (
                f"Since they are Not Yet ready, after presenting their position, naturally introduce "
                f"the alternative property as a smart stepping stone they could pursue now. "
                if alternative else ""
            )
            + f"Keep it to 4–5 sentences. Do not ask a question you already know the answer to."
        ),
    )
    st.session_state.l2_history = [
        {"role": "user", "text": "[property selected]"},
        {"role": "assistant", "text": opening},
    ]

    msgs = [
        {
            "role": "user",
            "type": "text",
            "content": f"I'd love to go with the {prop.get('ideal_buyer_archetype', 'property')} in {prop.get('suburb', 'Unknown')}.",
        },
        {"role": "assistant", "type": "text", "content": opening},
    ]

    # Show the alternative property card if one was found
    if alternative:
        alt_exp = f"At ${alternative.get('price', 0):,.0f}, this is within your current reach and matches your lifestyle."
        msgs.append({
            "role": "assistant",
            "type": "properties",
            "content": {
                "intro": "💡 **A property within your reach right now:**",
                "props": [alternative],
                "explanations": [alt_exp],
                "prompt": "",
            },
        })

    return msgs


def handle_layer2(user_input: str, client) -> List[Dict]:
    system = layer2_prompt(
        st.session_state.selected,
        st.session_state.profile,
        st.session_state.affordability,
        st.session_state.scenarios,
        st.session_state.alternative_property,
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

    user_names = financial_engine.get_all_user_names()
    uid_list = sorted(user_names.keys())
    name_list = [user_names[uid] for uid in uid_list]

    # Sidebar
    with st.sidebar:
        st.title("🏠 Occubuy")
        st.caption("AI home buying assistant")
        st.divider()

        current_idx = uid_list.index(st.session_state.user_id) if st.session_state.user_id in uid_list else 0
        selected_idx = st.selectbox(
            "Demo persona:",
            range(len(name_list)),
            format_func=lambda i: name_list[i],
            index=current_idx,
        )
        new_uid = uid_list[selected_idx]
        if new_uid != st.session_state.user_id:
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.user_id = new_uid
            st.rerun()

        st.divider()
        stage_labels = {
            "layer1": "🏠 Layer 1: Discovering your home",
            "selecting": "🏠 Layer 1: Choose your home",
            "layer2": "💰 Layer 2: Financial readiness",
        }
        st.info(stage_labels.get(st.session_state.stage, ""))

        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            uid = st.session_state.user_id
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.user_id = uid
            st.rerun()

        if not client:
            st.error("⚠️ OPENAI_API_KEY not set")

    # Seed welcome message using the user's name
    if not st.session_state.messages:
        name = financial_engine.get_user_name(st.session_state.user_id)
        first = name.split()[0]
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": (
                f"👋 Hi {first}! I'm **Occubuy** — your home buying assistant.\n\n"
                "Before we look at anything, I want to get to know you a little. "
                "The best home isn't just about rooms and price — it's about the life you want to live.\n\n"
                f"**So let's start simply: what does your life look like right now, "
                f"and what are you hoping to change or create with a new home?**"
            ),
        })

    # Render all messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_msg(msg)

    # Property selection buttons
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
                    with st.spinner("Pulling up your financial picture..."):
                        new_msgs = select_property(i, client, financial_engine, matching_engine)
                    for m in new_msgs:
                        st.session_state.messages.append(m)
                    st.rerun()

    # Chat input
    if st.session_state.stage != "selecting":
        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
            name = financial_engine.get_user_name(st.session_state.user_id)

            with st.spinner("..."):
                if st.session_state.stage == "layer1":
                    new_msgs = handle_layer1(prompt, client, rag_retriever, matching_engine, name)
                elif st.session_state.stage == "layer2":
                    new_msgs = handle_layer2(prompt, client)
                else:
                    new_msgs = []

            for m in new_msgs:
                st.session_state.messages.append(m)
            st.rerun()


if __name__ == "__main__":
    main()

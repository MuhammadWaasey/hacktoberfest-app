"""
Pakistani Medicine Price Finder - Streamlit + Gemma 4 (Gemini API)

Run:
    pip install streamlit google-genai python-dotenv
    streamlit run app.py
"""

import json
import re
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------------------------------
# Configuration & Setup
# --------------------------------------------------------------------------
load_dotenv()
MODEL = "gemma-4-26b-a4b-it"

st.set_page_config(
    page_title="Pakistani Medicine Price Finder",
    page_icon="💊",
    layout="centered",
)

# Custom Styling
st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 2rem;}
    .card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
        background: rgba(128,128,128,0.06);
    }
    .card h4 {margin: 0 0 .25rem 0;}
    .muted {opacity: .7; font-size: .85rem;}
    .price {font-size: 1.6rem; font-weight: 700; margin: .3rem 0;}
    .save {color: #1a9e55; font-weight: 600;}
    .nosave {color: #c0392b; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client():
    return genai.Client()


PROMPT_TEMPLATE = """You are a knowledgeable assistant on the Pakistani pharmaceutical market.

The user entered this medicine name: "{name}"

Identify it as a brand sold in Pakistan (or a generic name). Then give 3 CHEAPER alternative
brands available in Pakistan that contain the SAME active ingredient and the SAME strength
and dosage form. Prices must be realistic ESTIMATES in Pakistani Rupees (PKR) for a
comparable pack size (e.g. one strip / one bottle).

Respond ONLY with valid JSON in this exact structure, no markdown codeblocks:
{{
  "found": true,
  "medicine_name": "Properly spelled brand/generic name",
  "active_ingredient": "e.g. Paracetamol",
  "strength": "e.g. 500 mg",
  "dosage_form": "e.g. Tablet",
  "pack_size": "e.g. 1 strip of 10 tablets",
  "used_for": "one short plain-language sentence",
  "original_price_pkr": 0,
  "alternatives": [
    {{
      "brand": "Brand name",
      "manufacturer": "Company",
      "strength": "same strength",
      "pack_size": "pack size comparable to the original",
      "price_pkr": 0
    }}
  ]
}}

Rules:
- "alternatives" must contain 3 items sorted from cheapest to most expensive, each priced lower than original_price_pkr.
- All prices are plain numbers (no currency symbols).
- If input is not a medicine, return {{"found": false}}.
"""


def extract_json(text: str):
    """Safely extract JSON from model text response."""
    text = text.strip()
    # Strip markdown backticks if returned
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])


@st.cache_data(show_spinner=False, ttl=3600)
def lookup(name: str):
    client = get_client()
    
    # Retry loop for resilience against temporary 500 server errors
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=PROMPT_TEMPLATE.format(name=name),
            )
            return extract_json(response.text)
        except Exception as e:
            last_error = e
            time.sleep(1)  # Brief wait before retry
            
    raise last_error


def to_float(value) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def pkr(value: float) -> str:
    return f"Rs. {value:,.0f}"


# --------------------------------------------------------------------------
# UI Interface
# --------------------------------------------------------------------------
st.title("💊 Pakistani Medicine Price Finder")
st.caption("Find cheaper brand alternatives with the same active ingredient, in PKR.")

with st.form("search_form", clear_on_submit=False):
    col_in, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_in:
        medicine = st.text_input(
            "Medicine name",
            placeholder="e.g. Panadol, Augmentin, Brufen, Risek",
        )
    with col_btn:
        submitted = st.form_submit_button("Find prices", use_container_width=True, type="primary")

if submitted:
    name = medicine.strip()
    if not name:
        st.warning("Please enter a medicine name.")
        st.stop()

    try:
        with st.spinner(f"Looking up alternatives for {name}..."):
            data = lookup(name)
    except Exception as e:
        st.error(f"Error contacting model or parsing data: {e}")
        st.info("Tip: Click 'Find prices' again to retry.")
        st.stop()

    if not data.get("found", False):
        st.warning("That doesn't look like a recognisable medicine. Please check the spelling and try again.")
        st.stop()

    original_price = to_float(data.get("original_price_pkr"))
    alts = data.get("alternatives", [])[:3]

    # ---- 1. Active Ingredient ----
    st.subheader("1. Active Ingredient & Strength")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='card'><div class='muted'>Medicine</div>"
            f"<h4>{data.get('medicine_name', name)}</h4>"
            f"<div class='muted'>{data.get('dosage_form', '')}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card'><div class='muted'>Active ingredient</div>"
            f"<h4>{data.get('active_ingredient', '-')}</h4>"
            f"<div class='muted'>Strength: {data.get('strength', '-')}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='card'><div class='muted'>Estimated price</div>"
            f"<div class='price'>{pkr(original_price)}</div>"
            f"<div class='muted'>{data.get('pack_size', '')}</div></div>",
            unsafe_allow_html=True,
        )
    if data.get("used_for"):
        st.caption(f"Commonly used for: {data['used_for']}")

    # ---- 2. Cheaper Alternatives ----
    st.subheader("2. Cheaper Pakistani Brand Alternatives")
    if not alts:
        st.info("No cheaper alternatives were found for this medicine.")
    else:
        cols = st.columns(len(alts))
        for col, alt in zip(cols, alts):
            price = to_float(alt.get("price_pkr"))
            saving = original_price - price
            pct = (saving / original_price * 100) if original_price else 0
            
            save_html = (
                f"<div class='save'>Save {pkr(saving)} ({pct:.0f}%)</div>"
                if saving > 0
                else "<div class='nosave'>No saving</div>"
            )
            
            with col:
                st.markdown(
                    f"<div class='card'>"
                    f"<h4>{alt.get('brand', '-')}</h4>"
                    f"<div class='muted'>{alt.get('manufacturer', '')}</div>"
                    f"<div class='price'>{pkr(price)}</div>"
                    f"<div class='muted'>{alt.get('strength', '')} &middot; {alt.get('pack_size', '')}</div>"
                    f"{save_html}</div>",
                    unsafe_allow_html=True,
                )

        # ---- 3. Potential Savings ----
        st.subheader("3. Potential Savings")
        priced = [(a, to_float(a.get("price_pkr"))) for a in alts]
        best, best_price = min(priced, key=lambda x: x[1])
        best_saving = original_price - best_price
        best_pct = (best_saving / original_price * 100) if original_price else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Original price", pkr(original_price))
        m2.metric(f"Cheapest: {best.get('brand', '-')}", pkr(best_price))
        m3.metric("You save", pkr(max(best_saving, 0)), f"{best_pct:.0f}%" if best_saving > 0 else None)

        if best_saving > 0:
            st.success(
                f"Switching to **{best.get('brand')}** could save about **{pkr(best_saving)}** per pack "
                f"({best_pct:.0f}%). Over 12 packs, that's roughly **{pkr(best_saving * 12)}**."
            )

    # ---- 4. Disclaimer ----
    st.subheader("4. Important Disclaimer")
    st.warning(
        "**This information is for general guidance only and is not medical advice.** "
        "Prices are AI-generated estimates and may differ from actual retail prices. "
        "**Always consult a doctor or pharmacist before substituting any medicine.**"
    )
else:
    st.info("Enter a medicine name above and click **Find prices** to get started.")

st.markdown("---")
st.caption(f"Powered by {MODEL} via the Gemini API. Prices are estimates in PKR.")
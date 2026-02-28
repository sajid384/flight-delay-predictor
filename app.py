import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ─────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Custom CSS – dark aviation theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --navy:   #0a0f1e;
    --dark:   #0d1526;
    --card:   #111c33;
    --border: #1e3358;
    --blue:   #1a6bff;
    --cyan:   #00d4ff;
    --amber:  #ffb300;
    --red:    #ff4444;
    --green:  #00e676;
    --muted:  #5a7aab;
    --text:   #ccddf5;
}

/* ── Global background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Hide default toolbar padding ── */
.block-container { padding-top: 2rem !important; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    background: radial-gradient(ellipse at 50% 0%, #0f2a5e 0%, transparent 70%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-icon {
    font-size: 3.5rem;
    display: block;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0);   }
    50%      { transform: translateY(-8px); }
}
.hero h1 {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #fff 30%, var(--cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.4rem 0 0.6rem;
    letter-spacing: 0.04em;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Card wrapper ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #0d1526 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: #fff !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(26,107,255,0.25) !important;
}
[data-baseweb="select"] { background: #0d1526 !important; }
[data-baseweb="popover"] ul { background: #111c33 !important; }

/* Slider track colour */
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBarMin"] { color: var(--muted); }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--blue) 0%, #0052cc 100%);
    color: #fff;
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(26,107,255,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(26,107,255,0.55);
    background: linear-gradient(135deg, #2a7bff 0%, #0062dd 100%);
}

/* ── Result cards ── */
.result-on-time {
    background: linear-gradient(135deg, rgba(0,230,118,0.1), rgba(0,200,100,0.05));
    border: 1px solid rgba(0,230,118,0.4);
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-delayed {
    background: linear-gradient(135deg, rgba(255,68,68,0.12), rgba(200,0,0,0.05));
    border: 1px solid rgba(255,68,68,0.4);
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-emoji  { font-size: 3rem; display: block; margin-bottom: 0.4rem; }
.result-label  { font-family: 'Orbitron', monospace; font-size: 1.4rem; font-weight: 700; }
.result-prob   { font-size: 0.9rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Probability meter ── */
.prob-bar-bg {
    background: #1a2840;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.prob-bar-fill-green {
    height: 100%;
    background: linear-gradient(90deg, var(--green), #00b050);
    border-radius: 999px;
    transition: width 0.6s ease;
}
.prob-bar-fill-red {
    height: 100%;
    background: linear-gradient(90deg, var(--amber), var(--red));
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Feature importance table ── */
.fi-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}
.fi-rank {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    width: 1.6rem;
    text-align: center;
}
.fi-name { flex: 1; font-size: 0.9rem; color: var(--text); }
.fi-bar-bg { width: 120px; background: #1a2840; border-radius: 999px; height: 6px; overflow: hidden; }
.fi-bar-fill { height: 100%; background: linear-gradient(90deg, var(--blue), var(--cyan)); border-radius: 999px; }
.fi-val { font-size: 0.78rem; color: var(--muted); width: 3rem; text-align: right; }

/* ── Stats strip ── */
.stat-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--cyan);
}
.stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    letter-spacing: 0.06em;
}

/* hide streamlit built-in alerts */
div[data-testid="stAlert"] { display: none; }

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Load model  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("flight_delay_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, features = load_model()

# ─────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-icon">✈️</span>
  <h1>Flight Delay Predictor</h1>
  <p>AI-powered prediction · Real-time analysis · Departure intelligence</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Layout: inputs (left)  |  results (right)
# ─────────────────────────────────────────────
left_col, gap_col, right_col = st.columns([5, 0.5, 5])

# ───── LEFT: inputs ─────
with left_col:
    st.markdown('<div class="section-label">✦ Flight Parameters</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Row 1 – Month & Day
        c1, c2 = st.columns(2)
        with c1:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            month_label = st.selectbox(
                "📅  Departure Month",
                options=list(range(1, 13)),
                format_func=lambda x: f"{month_names[x-1]} ({x})",
            )
            month = month_label   # keeps same variable name as original

        with c2:
            day_names = ["Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday"]
            day_label = st.selectbox(
                "📆  Day of Week",
                options=list(range(1, 8)),
                format_func=lambda x: day_names[x-1],
            )
            day_of_week = day_label

        st.markdown("</div>", unsafe_allow_html=True)   # close first logical group

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Row 2 – Distance slider
        distance = st.slider(
            "🛫  Route Distance (miles)",
            min_value=50, max_value=5_500,
            value=1_000, step=25,
            help="Great-circle distance between origin and destination airports",
        )

        # Row 3 – Taxi out slider
        taxi_out = st.slider(
            "🕐  Taxi-Out Time (minutes)",
            min_value=1, max_value=90,
            value=15, step=1,
            help="Time from gate push-back to wheels-up",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Airline selectbox
        airline_codes = sorted(
            [col.replace("OP_UNIQUE_CARRIER_", "")
             for col in features if "OP_UNIQUE_CARRIER_" in col]
        )
        # Nice display names where known
        airline_display = {
            "AA":"American Airlines (AA)", "DL":"Delta Air Lines (DL)",
            "UA":"United Airlines (UA)",   "WN":"Southwest (WN)",
            "B6":"JetBlue (B6)",           "AS":"Alaska Airlines (AS)",
            "NK":"Spirit Airlines (NK)",   "F9":"Frontier (F9)",
            "G4":"Allegiant (G4)",         "HA":"Hawaiian Airlines (HA)",
        }
        airline = st.selectbox(
            "🛩️  Airline Carrier",
            options=airline_codes,
            format_func=lambda x: airline_display.get(x, x),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Stats strip
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value">{distance:,}</div>
            <div class="stat-label">Miles</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value">{taxi_out}</div>
            <div class="stat-label">Taxi-out min</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value">{airline}</div>
            <div class="stat-label">Carrier</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  ANALYZE FLIGHT", use_container_width=True)

# ───── RIGHT: results ─────
with right_col:
    st.markdown('<div class="section-label">✦ Prediction Results</div>', unsafe_allow_html=True)

    if not predict_btn:
        # Placeholder state
        st.markdown("""
        <div class="card" style="min-height:340px; display:flex; flex-direction:column;
             align-items:center; justify-content:center; gap:1rem; opacity:0.5;">
          <span style="font-size:3rem;">🛫</span>
          <p style="text-align:center; color:var(--muted); font-size:0.9rem; line-height:1.6;">
            Configure your flight parameters on the left<br>and click <strong>Analyze Flight</strong>
            to see the delay prediction.
          </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Build feature vector (original logic, untouched) ──────────────────
        new_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        new_data['MONTH']       = month
        new_data['DAY_OF_WEEK'] = day_of_week
        new_data['DISTANCE']    = distance
        new_data['TAXI_OUT']    = taxi_out

        airline_col = f"OP_UNIQUE_CARRIER_{airline}"
        if airline_col in new_data.columns:
            new_data[airline_col] = 1

        prediction  = model.predict(new_data)[0]
        proba_all   = model.predict_proba(new_data)
        proba_all   = np.atleast_2d(proba_all)
        probability = proba_all[0, int(prediction)]
        # ─────────────────────────────────────────────────────────────────────

        pct = int(probability * 100)

        if prediction == 0:
            st.markdown(f"""
            <div class="result-on-time">
              <span class="result-emoji">✅</span>
              <div class="result-label" style="color:var(--green)">ON TIME</div>
              <div class="result-prob">Model confidence: {pct}%</div>
              <div class="prob-bar-bg" style="margin-top:1rem">
                <div class="prob-bar-fill-green" style="width:{pct}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            verdict_color = "var(--green)"
        else:
            st.markdown(f"""
            <div class="result-delayed">
              <span class="result-emoji">⚠️</span>
              <div class="result-label" style="color:var(--red)">DELAY EXPECTED</div>
              <div class="result-prob">Model confidence: {pct}%</div>
              <div class="prob-bar-bg" style="margin-top:1rem">
                <div class="prob-bar-fill-red" style="width:{pct}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence gauge numbers ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        on_time_pct  = proba_all[0, 0] * 100
        delayed_pct  = proba_all[0, 1] * 100 if proba_all.shape[1] > 1 else (100 - on_time_pct)

        with g1:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-value" style="color:var(--green)">{on_time_pct:.0f}%</div>
                <div class="stat-label">On-time prob.</div></div>""", unsafe_allow_html=True)
        with g2:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-value" style="color:var(--red)">{delayed_pct:.0f}%</div>
                <div class="stat-label">Delay prob.</div></div>""", unsafe_allow_html=True)
        with g3:
            risk = "LOW" if prediction == 0 else "HIGH"
            risk_color = "var(--green)" if prediction == 0 else "var(--red)"
            st.markdown(f"""<div class="stat-box">
                <div class="stat-value" style="color:{risk_color}">{risk}</div>
                <div class="stat-label">Risk level</div></div>""", unsafe_allow_html=True)

        # ── Feature importance (original logic, untouched) ───────────────────
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fi = pd.Series(importances, index=features).sort_values(ascending=False).head(5)
            max_val = fi.values[0]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">✦ Top Influencing Features</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)

            rows_html = ""
            for i, (feat, val) in enumerate(fi.items(), 1):
                bar_pct = int((val / max_val) * 100)
                clean   = feat.replace("OP_UNIQUE_CARRIER_", "Carrier: ").replace("_", " ").title()
                rows_html += f"""
                <div class="fi-row">
                  <span class="fi-rank">#{i}</span>
                  <span class="fi-name">{clean}</span>
                  <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:{bar_pct}%"></div></div>
                  <span class="fi-val">{val:.3f}</span>
                </div>"""

            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("""
<hr>
<div class="footer">
  ✈️ &nbsp;Flight Delay Predictor &nbsp;·&nbsp; Powered by Machine Learning
  &nbsp;·&nbsp; Predictions are probabilistic and for informational purposes only
</div>
""", unsafe_allow_html=True)
"""
Shared visual theme for the Streamlit app: the sentiment color palette (single
source of truth, used by every chart/badge in the app) and one function that
injects the "modern SaaS dashboard" CSS look - dark sidebar, card-style stat
tiles, a compact gradient header, restyled tabs/buttons/tables.

Only our own static CSS/HTML strings are ever passed through
unsafe_allow_html here - never user-supplied comment text.
"""
import streamlit as st

PRIMARY = "#4F46E5"
PRIMARY_DARK = "#4338CA"
ACCENT = "#7C3AED"

SENTIMENT_COLORS = {"Positive": "#16A34A", "Negative": "#DC2626", "Neutral": "#64748B"}
SENTIMENT_COLORMAPS = {"Positive": "Greens", "Negative": "Reds", "Neutral": "Greys"}
SENTIMENT_BG = {"Positive": "#F0FDF4", "Negative": "#FEF2F2", "Neutral": "#F8FAFC"}


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}

        /* ---- Sidebar: dark panel ---- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #E2E8F0 !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: #334155 !important;
        }}

        /* ---- Compact gradient hero header ---- */
        .hero-header {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {ACCENT} 100%);
            border-radius: 16px;
            padding: 22px 28px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
        }}
        .hero-header h1 {{
            color: white !important;
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0 0 4px 0;
            line-height: 1.25;
        }}
        .hero-header p {{
            color: rgba(255,255,255,0.9) !important;
            font-size: 0.92rem;
            margin: 0;
        }}

        /* ---- Card-style st.metric tiles ---- */
        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }}
        [data-testid="stMetricLabel"] {{
            font-weight: 600;
            color: #64748B !important;
        }}

        /* ---- Generic reusable stat/pipeline cards (our own HTML) ---- */
        .stat-card {{
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            text-align: center;
            height: 100%;
        }}
        .stat-card .big {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {PRIMARY_DARK};
            margin: 4px 0;
        }}
        .stat-card .label {{
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .stat-card .icon {{
            font-size: 1.6rem;
        }}

        .pipeline-card {{
            background: white;
            border: 1px solid #E2E8F0;
            border-left: 4px solid {PRIMARY};
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }}
        .pipeline-card .stage {{
            font-size: 0.78rem;
            color: #64748B;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .pipeline-card .title {{
            font-weight: 700;
            color: #0F172A;
            margin: 2px 0;
        }}
        .pipeline-card .detail {{
            font-size: 0.85rem;
            color: #475569;
        }}

        .result-card {{
            border-radius: 14px;
            padding: 18px 20px;
            margin-top: 10px;
            border: 1px solid rgba(0,0,0,0.06);
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-weight: 600;
            padding: 10px 16px;
        }}

        /* ---- Buttons ---- */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {ACCENT} 100%);
            border: none;
        }}

        /* ---- Dataframes ---- */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def stat_card(icon: str, value: str, label: str) -> str:
    return (
        f'<div class="stat-card"><div class="icon">{icon}</div>'
        f'<div class="big">{value}</div><div class="label">{label}</div></div>'
    )


def pipeline_card(stage: str, title: str, detail: str) -> str:
    return (
        f'<div class="pipeline-card"><div class="stage">{stage}</div>'
        f'<div class="title">{title}</div><div class="detail">{detail}</div></div>'
    )

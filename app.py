import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SyncMatch · AI Resume Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #0a0a0f !important;
    color: #e8e6f0 !important;
}
.stApp { background: #0a0a0f !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2affd0; border-radius: 4px; }

.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(42,255,208,0.12) 0%, transparent 70%);
    border-bottom: 1px solid rgba(42,255,208,0.1);
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(42,255,208,0.08);
    border: 1px solid rgba(42,255,208,0.3);
    color: #2affd0;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3.4rem) !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.03em !important;
    margin: 0.2rem 0 1rem !important;
    background: linear-gradient(135deg, #ffffff 30%, #2affd0 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
.hero p { color: #8b899e; font-size: 1rem; max-width: 520px; margin: 0 auto; line-height: 1.6; }

.upload-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: rgba(42,255,208,0.25); }
.card-label {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2affd0;
    margin-bottom: 0.8rem;
}

.stFileUploader > div > div {
    background: #0d0d18 !important;
    border: 1.5px dashed #2a2a3e !important;
    border-radius: 12px !important;
}
.stFileUploader > div > div:hover { border-color: #2affd0 !important; }

.stTextArea textarea {
    background: #0d0d18 !important;
    border: 1.5px solid #1e1e2e !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    min-height: 180px !important;
}
.stTextArea textarea:focus { border-color: #2affd0 !important; box-shadow: 0 0 0 3px rgba(42,255,208,0.08) !important; }

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2affd0 0%, #00c9a7 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 24px rgba(42,255,208,0.2) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(42,255,208,0.35) !important; }

.score-ring-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:2rem 1rem; }
.ring-svg { filter: drop-shadow(0 0 18px rgba(42,255,208,0.35)); }

.metric-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card .m-label { font-size:0.65rem; font-family:'DM Mono',monospace; letter-spacing:0.16em; text-transform:uppercase; color:#55536a; margin-bottom:0.5rem; }
.metric-card .m-value { font-size:2rem; font-weight:800; letter-spacing:-0.03em; color:#2affd0; }
.metric-card .m-sub { font-size:0.75rem; color:#55536a; margin-top:0.3rem; }

.section-header { display:flex; align-items:center; gap:0.7rem; margin:2.2rem 0 1.2rem; padding-bottom:0.7rem; border-bottom:1px solid #1e1e2e; }
.section-header .sh-icon { width:28px;height:28px; background:rgba(42,255,208,0.1); border-radius:8px; display:flex;align-items:center;justify-content:center; font-size:0.9rem; }
.section-header .sh-title { font-size:1rem; font-weight:700; letter-spacing:-0.01em; color:#e8e6f0; }

.pill-wrap { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.8rem; }
.pill { font-family:'DM Mono',monospace; font-size:0.72rem; padding:0.3rem 0.8rem; border-radius:100px; letter-spacing:0.02em; font-weight:500; }
.pill-match { background:rgba(42,255,208,0.1); border:1px solid rgba(42,255,208,0.3); color:#2affd0; }
.pill-miss  { background:rgba(255,80,100,0.08); border:1px solid rgba(255,80,100,0.25); color:#ff7088; }
.pill-extra { background:rgba(168,139,255,0.08); border:1px solid rgba(168,139,255,0.25); color:#a88bff; }

.sug-card { background:#13131f; border:1px solid #1e1e2e; border-left:3px solid #2affd0; border-radius:12px; padding:1rem 1.3rem; margin-bottom:0.8rem; font-size:0.88rem; line-height:1.6; color:#c0bdd4; }
.sug-card.warn  { border-left-color:#ffb347; }
.sug-card.error { border-left-color:#ff5064; }
.sug-card strong { color:#e8e6f0; }

.verdict { border-radius:14px; padding:1.2rem 1.6rem; display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem; }
.verdict.v-success { background:rgba(42,255,208,0.07); border:1px solid rgba(42,255,208,0.2); }
.verdict.v-info    { background:rgba(168,139,255,0.07); border:1px solid rgba(168,139,255,0.2); }
.verdict.v-error   { background:rgba(255,80,100,0.07);  border:1px solid rgba(255,80,100,0.2); }
.verdict .v-emoji  { font-size:1.8rem; }
.verdict .v-text   { font-size:0.9rem; color:#c0bdd4; line-height:1.5; }
.verdict .v-text strong { font-size:1rem; color:#e8e6f0; display:block; margin-bottom:0.15rem; }

hr { border-color:#1e1e2e !important; margin:1.5rem 0 !important; }
.stSpinner > div { border-top-color:#2affd0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SKILL DICTIONARY
# ─────────────────────────────────────────────
SKILL_DICTIONARY = {
    "python", "r", "sql", "java", "scala", "julia",
    "c++", "c#", "javascript", "typescript", "bash", "shell",
    "mysql", "postgresql", "sqlite", "mongodb", "cassandra",
    "redis", "bigquery", "snowflake", "redshift", "oracle",
    "sql server", "ms sql", "hive", "presto", "athena",
    "power bi", "tableau", "looker", "metabase", "qlik",
    "matplotlib", "seaborn", "plotly", "ggplot", "d3.js",
    "excel", "google sheets", "data studio", "looker studio",
    "eda", "exploratory data analysis",
    "data cleaning", "data wrangling", "data transformation",
    "data modeling", "data pipeline", "data warehousing",
    "etl", "elt", "data engineering",
    "feature engineering", "feature selection",
    "statistical analysis", "regression analysis",
    "hypothesis testing", "a/b testing",
    "cohort analysis", "funnel analysis", "churn analysis",
    "kpi", "metrics", "business intelligence",
    "dashboarding", "reporting", "data visualization",
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "supervised learning", "unsupervised learning",
    "linear regression", "logistic regression",
    "decision tree", "random forest", "xgboost", "lightgbm",
    "gradient boosting", "neural network",
    "classification", "clustering", "dimensionality reduction",
    "pca", "k-means", "svm", "naive bayes",
    "model evaluation", "cross validation", "hyperparameter tuning",
    "mlops", "model deployment",
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
    "hugging face", "transformers", "spacy", "nltk",
    "pandas", "numpy", "scipy", "statsmodels",
    "aws", "azure", "gcp", "google cloud",
    "s3", "ec2", "lambda", "sagemaker",
    "databricks", "dataflow",
    "spark", "apache spark", "hadoop", "kafka",
    "airflow", "dbt", "luigi", "flink",
    "git", "github", "gitlab", "docker", "kubernetes",
    "ci/cd", "jenkins", "jira", "confluence",
    "product analytics", "user research", "ux research",
    "requirement gathering", "business analysis",
    "stakeholder management", "roadmapping",
    "agile", "scrum", "kanban", "sprint planning",
    "wireframing", "prototyping", "figma",
    "customer journey", "persona mapping",
    "market research", "competitive analysis",
    "statistics", "probability", "bayesian",
    "time series", "forecasting", "arima",
    "communication", "presentation", "problem solving",
}
SORTED_SKILLS = sorted(SKILL_DICTIONARY, key=len, reverse=True)


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "
    return text

def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-/#+]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text: str) -> set:
    norm = normalise(text)
    found = set()
    for skill in SORTED_SKILLS:
        if any(c in skill for c in (" ", "/", "-")):
            if skill in norm:
                found.add(skill)
        else:
            if re.search(r"\b" + re.escape(skill) + r"\b", norm):
                found.add(skill)
    return found

def compute_similarity(t1: str, t2: str) -> float:
    vec = TfidfVectorizer()
    v = vec.fit_transform([normalise(t1), normalise(t2)])
    return float(cosine_similarity(v[0], v[1])[0][0])

def ats_score(match_pct, skill_pct):
    return min(round(match_pct * 0.40 + skill_pct * 0.50 + 10, 1), 100.0)


# ─────────────────────────────────────────────
# HTML COMPONENT HELPERS
# ─────────────────────────────────────────────
def pills_html(skills, pill_class):
    if not skills:
        return "<p style='color:#55536a;font-size:0.8rem;margin-top:0.5rem;'>None detected.</p>"
    pills = "".join(f'<span class="pill {pill_class}">{s}</span>' for s in skills)
    return f'<div class="pill-wrap">{pills}</div>'

def score_ring_svg(score):
    r = 70
    circum = 2 * 3.14159 * r
    filled = circum * (score / 100)
    gap    = circum - filled
    color  = "#2affd0" if score >= 80 else ("#a88bff" if score >= 60 else "#ff5064")
    return f"""
    <div class="score-ring-wrap">
      <svg class="ring-svg" width="180" height="180" viewBox="0 0 180 180">
        <circle cx="90" cy="90" r="{r}" fill="none" stroke="#1e1e2e" stroke-width="10"/>
        <circle cx="90" cy="90" r="{r}" fill="none" stroke="{color}" stroke-width="10"
          stroke-linecap="round"
          stroke-dasharray="{filled:.1f} {gap:.1f}"
          transform="rotate(-90 90 90)"/>
        <text x="90" y="84" text-anchor="middle"
          font-family="Syne,sans-serif" font-size="28" font-weight="800" fill="{color}">{score}</text>
        <text x="90" y="104" text-anchor="middle"
          font-family="DM Mono,monospace" font-size="10" fill="#55536a" letter-spacing="2">ATS SCORE</text>
      </svg>
      <p style="font-size:0.7rem;font-family:'DM Mono',monospace;letter-spacing:0.12em;
                text-transform:uppercase;color:#55536a;margin-top:0.3rem;">out of 100</p>
    </div>"""

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
      <div class="m-label">{label}</div>
      <div class="m-value">{value}</div>
      {'<div class="m-sub">'+sub+'</div>' if sub else ''}
    </div>"""

def section_header(icon, title):
    return f"""<div class="section-header">
      <div class="sh-icon">{icon}</div>
      <div class="sh-title">{title}</div>
    </div>"""

def sug_card(text, level=""):
    return f'<div class="sug-card {level}">{text}</div>'

def verdict(emoji, heading, body, level):
    return f"""<div class="verdict v-{level}">
      <div class="v-emoji">{emoji}</div>
      <div class="v-text"><strong>{heading}</strong>{body}</div>
    </div>"""

def col_label(icon, text, color):
    return f"""<p style="font-size:0.7rem;font-family:'DM Mono',monospace;letter-spacing:0.14em;
                text-transform:uppercase;color:{color};margin-bottom:0.2rem;">{icon} &nbsp;{text}</p>"""


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ AI-Powered · NLP · TF-IDF</div>
  <h1>Resume ↔ JD Matcher</h1>
  <p>Upload your resume and paste a job description. Get a real skills-based ATS score with zero fluff.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="upload-card"><div class="card-label">📄 &nbsp;Your Resume (PDF)</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="upload-card"><div class="card-label">📋 &nbsp;Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area("", height=180, placeholder="Paste the full job description here…", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("⚡  Analyze My Resume")


# ─────────────────────────────────────────────
# ANALYSIS & RESULTS
# ─────────────────────────────────────────────
if analyze:
    if resume_file is None or not jd_text.strip():
        st.warning("⚠️  Please upload a resume PDF and paste a job description.")
        st.stop()

    with st.spinner("Crunching your resume against the JD…"):
        resume_text     = extract_text_from_pdf(resume_file)
        similarity      = compute_similarity(resume_text, jd_text)
        match_pct       = round(similarity * 100, 1)
        resume_skills   = extract_skills(resume_text)
        jd_skills       = extract_skills(jd_text)
        matching_skills = sorted(resume_skills & jd_skills)
        missing_skills  = sorted(jd_skills - resume_skills)
        extra_skills    = sorted(resume_skills - jd_skills)
        skill_pct       = round((len(matching_skills) / len(jd_skills)) * 100, 1) if jd_skills else 0.0
        final_ats       = ats_score(match_pct, skill_pct)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Score ring + metrics ──────────────────
    ring_col, metrics_col = st.columns([1, 2], gap="large")

    with ring_col:
        st.markdown(score_ring_svg(final_ats), unsafe_allow_html=True)

    with metrics_col:
        st.markdown(section_header("📊", "Score Breakdown"), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(metric_card("Match Score",   f"{match_pct}%",        "TF-IDF similarity"), unsafe_allow_html=True)
        with m2: st.markdown(metric_card("Skills Match",  f"{skill_pct}%",        f"{len(matching_skills)} of {len(jd_skills)} skills"), unsafe_allow_html=True)
        with m3: st.markdown(metric_card("JD Skills",     str(len(jd_skills)),    "detected in JD"), unsafe_allow_html=True)

        if final_ats >= 80:
            st.markdown(verdict("🔥", "Excellent Match", "Your resume is strongly aligned. Apply with confidence.", "success"), unsafe_allow_html=True)
        elif final_ats >= 60:
            st.markdown(verdict("👍", "Good Match", "Solid profile — a few targeted additions can push you to the top tier.", "info"), unsafe_allow_html=True)
        else:
            st.markdown(verdict("⚠️", "Low Match", "Significant gaps detected. Tailor your resume before applying.", "error"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Skills breakdown ──────────────────────
    st.markdown(section_header("🔍", "Skills Breakdown"), unsafe_allow_html=True)
    sk1, sk2, sk3 = st.columns(3)

    with sk1:
        st.markdown(col_label("✅", "Matching Skills", "#2affd0"), unsafe_allow_html=True)
        st.markdown(pills_html(matching_skills, "pill-match"), unsafe_allow_html=True)

    with sk2:
        st.markdown(col_label("❌", "Missing Skills", "#ff7088"), unsafe_allow_html=True)
        st.markdown(pills_html(missing_skills, "pill-miss"), unsafe_allow_html=True)

    with sk3:
        st.markdown(col_label("➕", "Extra on Resume", "#a88bff"), unsafe_allow_html=True)
        st.markdown(pills_html(extra_skills[:12], "pill-extra"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Recommendations ───────────────────────
    st.markdown(section_header("💡", "Smart Recommendations"), unsafe_allow_html=True)

    if missing_skills:
        top = ", ".join(f"<strong>{s}</strong>" for s in missing_skills[:6])
        st.markdown(sug_card(
            f"🔧 <strong>Add these skills to your resume:</strong> {top}<br>"
            "<span style='font-size:0.8rem;'>Weave them into project bullets if you have hands-on experience.</span>"
        ), unsafe_allow_html=True)

    if match_pct < 55:
        st.markdown(sug_card(
            "📝 <strong>Mirror the JD's language.</strong> ATS engines do exact-phrase matching. "
            "If the JD says <em>'data pipeline'</em>, use that exact phrase — not <em>'data workflow'</em>.", "warn"
        ), unsafe_allow_html=True)

    if skill_pct < 40 and len(jd_skills) > 0:
        st.markdown(sug_card(
            "📚 <strong>Upskilling priority:</strong> Over 60% of required skills are absent. "
            "Pick the top 2–3 missing tools and build a quick project or get certified before applying.", "error"
        ), unsafe_allow_html=True)

    if final_ats >= 80:
        st.markdown(sug_card(
            "✨ <strong>You're in great shape.</strong> Customise your summary line to echo "
            "the job title and key stack — it nudges human reviewers as much as the ATS."
        ), unsafe_allow_html=True)

    if not jd_skills:
        st.markdown(sug_card(
            "⚠️ <strong>No skills detected in the JD.</strong> The description may be too generic. "
            "Try pasting a more detailed version for better results.", "warn"
        ), unsafe_allow_html=True)

    # ── Footer ────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
                border-top:1px solid #1e1e2e;color:#2a2a3e;font-size:0.72rem;
                font-family:'DM Mono',monospace;letter-spacing:0.1em;">
      BUILT WITH PYTHON · SKLEARN · PDFPLUMBER · STREAMLIT
    </div>
    """, unsafe_allow_html=True)

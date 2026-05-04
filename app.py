import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SyncMatch · AI Resume Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
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

/* ── Global text visibility — dark bg fix ── */
div, span, p, label, small, li { color: #c0bdd4; }
strong, b { color: #e8e6f0 !important; }
h1, h2, h3, h4, h5, h6 { color: #e8e6f0 !important; }
[data-testid="stMetricLabel"] { color: #8b899e !important; }
[data-testid="stMetricValue"] { color: #2affd0 !important; }
[data-testid="stNotification"] p, [data-testid="stAlert"] p { color: #e8e6f0 !important; }

.hero {
    text-align: center; padding: 3.5rem 1rem 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(42,255,208,0.12) 0%, transparent 70%);
    border-bottom: 1px solid rgba(42,255,208,0.1); margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block; background: rgba(42,255,208,0.08);
    border: 1px solid rgba(42,255,208,0.3); color: #2affd0;
    font-size: 0.7rem; font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.18em; text-transform: uppercase;
    padding: 0.3rem 0.9rem; border-radius: 100px; margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3.4rem) !important; font-weight: 800 !important;
    line-height: 1.1 !important; letter-spacing: -0.03em !important;
    margin: 0.2rem 0 1rem !important;
    background: linear-gradient(135deg, #ffffff 30%, #2affd0 100%);
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
}
.hero p { color: #8b899e; font-size: 1rem; max-width: 520px; margin: 0 auto; line-height: 1.6; }

.upload-card {
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 0.5rem;
}
.card-label {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #2affd0 !important;
    margin-bottom: 0.9rem;
    font-weight: 600;
    display: block;
}
/* ── File uploader box ───────────────────── */
.stFileUploader > div > div {
    background: #0d0d18 !important;
    border: 1.5px dashed #3a3a52 !important;
    border-radius: 12px !important;
}
.stFileUploader > div > div:hover { border-color: #2affd0 !important; }

/* Drag-drop instruction text & size hint */
.stFileUploader span,
.stFileUploader p,
.stFileUploader small,
.stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] span,
.stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] small,
.stFileUploader label,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p {
    color: #a09dba !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Upload button — clean restyle, no duplicate text */
.stFileUploader button {
    background: rgba(42,255,208,0.08) !important;
    border: 1px solid rgba(42,255,208,0.3) !important;
    color: #2affd0 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
/* This hides the extra injected span inside Streamlit's upload button that causes "upload Upload" */
.stFileUploader [data-testid="stFileUploaderDropzone"] button > div {
    display: none !important;
}
.stFileUploader [data-testid="stFileUploaderDropzone"] button::before {
    content: "Browse file";
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #2affd0;
}

/* Uploaded file name */
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
.uploadedFileName { color: #e8e6f0 !important; }

/* ── Textarea ────────────────────────────── */
.stTextArea textarea {
    background: #0d0d18 !important;
    border: 1.5px solid #2a2a3e !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #2affd0 !important;
    box-shadow: 0 0 0 3px rgba(42,255,208,0.08) !important;
}
/* Placeholder text — bright enough to read */
.stTextArea textarea::placeholder {
    color: #6b6882 !important;
    opacity: 1 !important;
    font-style: italic !important;
}

/* ── Any leftover Streamlit default text ─── */
.stMarkdown p, .stMarkdown span { color: #e8e6f0 !important; }
[data-testid="stText"] { color: #a09dba !important; }
p { color: #c0bdd4 !important; }
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2affd0 0%, #00c9a7 100%) !important;
    color: #0a0a0f !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    letter-spacing: 0.04em !important; border: none !important;
    border-radius: 12px !important; padding: 0.85rem 2rem !important;
    margin-top: 1rem !important; box-shadow: 0 4px 24px rgba(42,255,208,0.2) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(42,255,208,0.35) !important; }

.metric-card {
    background: #13131f; border: 1px solid #1e1e2e; border-radius: 14px;
    padding: 1.4rem 1.6rem; text-align: center; margin-bottom: 1rem;
}
.metric-card .m-label { font-size:0.65rem; font-family:'DM Mono',monospace; letter-spacing:0.16em; text-transform:uppercase; color:#55536a; margin-bottom:0.5rem; }
.metric-card .m-value { font-size:2rem; font-weight:800; letter-spacing:-0.03em; color:#2affd0; }
.metric-card .m-sub   { font-size:0.75rem; color:#55536a; margin-top:0.3rem; }

.section-header { display:flex; align-items:center; gap:0.7rem; margin:2.2rem 0 1.2rem; padding-bottom:0.7rem; border-bottom:1px solid #1e1e2e; }
.section-header .sh-icon { width:28px;height:28px; background:rgba(42,255,208,0.1); border-radius:8px; display:flex;align-items:center;justify-content:center; font-size:0.9rem; }
.section-header .sh-title { font-size:1rem; font-weight:700; color:#e8e6f0; }

.pill-wrap { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.8rem; }
.pill { font-family:'DM Mono',monospace; font-size:0.72rem; padding:0.3rem 0.8rem; border-radius:100px; font-weight:500; }
.pill-match { background:rgba(42,255,208,0.1); border:1px solid rgba(42,255,208,0.3); color:#2affd0; }
.pill-miss  { background:rgba(255,80,100,0.08); border:1px solid rgba(255,80,100,0.25); color:#ff7088; }
.pill-extra { background:rgba(168,139,255,0.08); border:1px solid rgba(168,139,255,0.25); color:#a88bff; }
.pill-high  { background:rgba(255,179,71,0.1); border:1px solid rgba(255,179,71,0.3); color:#ffb347; }

.verdict { border-radius:14px; padding:1.2rem 1.6rem; display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem; }
.verdict.v-success { background:rgba(42,255,208,0.07); border:1px solid rgba(42,255,208,0.2); }
.verdict.v-info    { background:rgba(168,139,255,0.07); border:1px solid rgba(168,139,255,0.2); }
.verdict.v-error   { background:rgba(255,80,100,0.07);  border:1px solid rgba(255,80,100,0.2); }
.verdict .v-emoji  { font-size:1.8rem; }
.verdict .v-text   { font-size:0.9rem; color:#c0bdd4; line-height:1.5; }
.verdict .v-text strong { font-size:1rem; color:#e8e6f0; display:block; margin-bottom:0.15rem; }

/* ── Fix card — the new key component ─────── */
.fix-card {
    background: #13131f; border: 1px solid #1e1e2e;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.fix-skill-badge {
    display: inline-block; background: rgba(255,80,100,0.1);
    border: 1px solid rgba(255,80,100,0.3); color: #ff7088;
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.2rem 0.7rem; border-radius: 100px; margin-bottom: 0.9rem;
}
.fix-skill-badge.high {
    background: rgba(255,179,71,0.1); border-color: rgba(255,179,71,0.3); color: #ffb347;
}
.fix-section-tag {
    display: inline-block; background: rgba(42,255,208,0.07);
    border: 1px solid rgba(42,255,208,0.2); color: #2affd0;
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    padding: 0.15rem 0.6rem; border-radius: 6px;
    margin-left: 0.5rem; vertical-align: middle;
}
.fix-label {
    font-size: 0.65rem; font-family: 'DM Mono', monospace;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 0.4rem; margin-top: 0.9rem;
}
.fix-label.old { color: #55536a; }
.fix-label.new { color: #2affd0; }
.fix-bullet-old {
    background: #0d0d18; border: 1px solid #1e1e2e; border-left: 3px solid #2a2a3e;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.83rem; color: #6b6882; line-height: 1.6;
    font-family: 'DM Mono', monospace;
}
.fix-bullet-new {
    background: rgba(42,255,208,0.04); border: 1px solid rgba(42,255,208,0.15);
    border-left: 3px solid #2affd0;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.83rem; color: #d0faf3; line-height: 1.6;
    font-family: 'DM Mono', monospace;
}
.fix-bullet-new mark {
    background: rgba(42,255,208,0.2); color: #2affd0;
    padding: 0.1rem 0.3rem; border-radius: 4px;
}
.fix-no-bullet {
    background: rgba(255,179,71,0.05); border: 1px solid rgba(255,179,71,0.2);
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #ffb347; line-height: 1.6;
}

/* ── Checklist ───────────────────────────── */
.check-row {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.6rem 0; border-bottom: 1px solid #1a1a28;
    font-size: 0.88rem;
}
.check-icon { font-size: 1rem; width: 22px; text-align: center; }
.check-found { color: #2affd0; }
.check-missing { color: #ff7088; }

hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }
.stSpinner > div { border-top-color: #2affd0 !important; }
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

# Section keywords for detection
SECTION_KEYWORDS = {
    "experience": ["experience", "work history", "employment", "internship", "intern", "trainee", "engineer", "analyst"],
    "skills":     ["skills", "technical skills", "core competencies", "tools", "technologies"],
    "projects":   ["project", "projects", "personal project", "academic project"],
    "education":  ["education", "qualification", "degree", "bachelor", "master", "b.tech", "m.tech"],
    "summary":    ["summary", "objective", "profile", "about me", "career objective"],
}

# Skill → related concept map for smarter bullet matching
SKILL_CONTEXT = {
    "tableau":        ["visualization", "dashboard", "report", "chart", "visual", "bi", "insight"],
    "power bi":       ["visualization", "dashboard", "report", "chart", "visual", "bi", "insight"],
    "scikit-learn":   ["model", "machine learning", "classification", "prediction", "algorithm", "sklearn"],
    "machine learning": ["model", "prediction", "algorithm", "classification", "data"],
    "sql":            ["database", "query", "data", "table", "mysql", "postgres"],
    "python":         ["script", "automation", "analysis", "data", "code"],
    "etl":            ["pipeline", "data", "transform", "load", "extract", "process"],
    "spark":          ["big data", "hadoop", "data processing", "pipeline"],
    "airflow":        ["pipeline", "workflow", "etl", "schedule", "orchestration"],
    "docker":         ["deployment", "container", "devops", "deploy"],
    "git":            ["version control", "code", "repository", "github"],
    "agile":          ["sprint", "scrum", "team", "project", "delivery"],
    "statistical analysis": ["statistics", "analysis", "test", "hypothesis", "data"],
    "a/b testing":    ["experiment", "test", "analysis", "conversion", "hypothesis"],
    "data cleaning":  ["data", "preprocessing", "wrangling", "null", "missing"],
}


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
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


def extract_bullets(text: str) -> list:
    """
    Pull every meaningful line from the resume that looks like a bullet / job description line.
    Filters out section headers, short lines, and contact info.
    """
    lines = text.split("\n")
    bullets = []
    for line in lines:
        line = line.strip()
        # Remove bullet characters
        line = re.sub(r"^[\•\-\*\➢\▪\◦\→]\s*", "", line)
        # Keep lines that are sentence-like (30–300 chars, contain a verb hint)
        if 30 <= len(line) <= 300:
            bullets.append(line)
    return bullets


def detect_section(bullet: str, full_text: str) -> str:
    """
    Find which resume section a bullet most likely belongs to
    by looking at what section header appears above it in the text.
    """
    lines = full_text.split("\n")
    bullet_idx = None
    for i, line in enumerate(lines):
        if bullet.lower()[:30] in line.lower():
            bullet_idx = i
            break

    if bullet_idx is None:
        return "Experience"

    # Walk backwards to find nearest section header
    for i in range(bullet_idx, max(0, bullet_idx - 20), -1):
        line_lower = lines[i].lower().strip()
        for section, keywords in SECTION_KEYWORDS.items():
            if any(kw in line_lower for kw in keywords):
                return section.title()

    return "Experience"


def rank_missing_skills(missing: set, jd_text: str) -> list:
    """
    Rank missing skills by how many times they appear in the JD.
    More mentions = higher priority.
    """
    jd_lower = jd_text.lower()
    ranked = []
    for skill in missing:
        count = len(re.findall(re.escape(skill), jd_lower))
        ranked.append((skill, count))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked  # list of (skill, jd_mention_count)


def find_best_bullet(skill: str, bullets: list) -> str | None:
    """
    Find the most relevant existing resume bullet to attach a missing skill to.
    Uses TF-IDF cosine similarity between the skill's context words and each bullet.
    """
    if not bullets:
        return None

    context_words = SKILL_CONTEXT.get(skill, [skill])
    query = skill + " " + " ".join(context_words)

    try:
        vec = TfidfVectorizer()
        corpus = [normalise(query)] + [normalise(b) for b in bullets]
        matrix = vec.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
        best_idx = sims.argmax()
        if sims[best_idx] > 0.05:   # minimum relevance threshold
            return bullets[best_idx]
    except Exception:
        pass
    return None


def suggest_rewrite(bullet: str, skill: str) -> str:
    """
    Generate a suggested rewrite of the bullet that naturally incorporates the missing skill.
    Uses simple rule-based insertion — no LLM required.
    """
    bullet = bullet.strip().rstrip(".")

    # Patterns for different skill types
    viz_skills    = {"tableau", "power bi", "looker", "matplotlib", "seaborn", "plotly", "data studio"}
    ml_skills     = {"scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm"}
    db_skills     = {"mysql", "postgresql", "mongodb", "bigquery", "snowflake", "redshift"}
    pipeline_skills = {"airflow", "etl", "spark", "kafka", "dbt"}
    cloud_skills  = {"aws", "azure", "gcp", "google cloud", "sagemaker"}

    skill_cap = skill.title() if len(skill) > 4 else skill.upper()

    if skill in viz_skills:
        return f"{bullet}, and visualized insights using <mark>{skill_cap}</mark> dashboards."
    elif skill in ml_skills:
        return f"{bullet}, leveraging <mark>{skill_cap}</mark> for model building and evaluation."
    elif skill in db_skills:
        return f"{bullet} using <mark>{skill_cap}</mark> for data extraction and querying."
    elif skill in pipeline_skills:
        return f"{bullet}, building automated data pipelines with <mark>{skill_cap}</mark>."
    elif skill in cloud_skills:
        return f"{bullet}, deployed and managed on <mark>{skill_cap}</mark>."
    elif skill == "git" or skill == "github":
        return f"{bullet}; maintained version control using <mark>{skill_cap}</mark>."
    elif skill == "agile" or skill == "scrum":
        return f"{bullet} following <mark>{skill_cap}</mark> methodology with sprint-based delivery."
    elif skill in {"statistical analysis", "hypothesis testing", "a/b testing"}:
        return f"{bullet}, applying <mark>{skill_cap}</mark> to validate findings."
    elif skill == "docker":
        return f"{bullet}, containerized using <mark>Docker</mark> for consistent deployment."
    else:
        return f"{bullet}, utilizing <mark>{skill_cap}</mark> to enhance outcomes."


def check_resume_sections(text: str) -> dict:
    """Check if key resume sections exist."""
    text_lower = text.lower()
    return {
        "Summary / Objective":  any(kw in text_lower for kw in ["summary", "objective", "profile", "about"]),
        "Skills Section":       any(kw in text_lower for kw in ["skills", "technical skills", "competencies", "tools"]),
        "Work Experience":      any(kw in text_lower for kw in ["experience", "work history", "employment", "internship"]),
        "Projects":             any(kw in text_lower for kw in ["project", "projects"]),
        "Education":            any(kw in text_lower for kw in ["education", "degree", "bachelor", "b.tech", "b.e"]),
        "Contact Info":         any(kw in text_lower for kw in ["email", "phone", "linkedin", "github", "@"]),
    }


# ─────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────
def pills_html(skills, pill_class):
    if not skills:
        return "<p style='color:#55536a;font-size:0.8rem;margin-top:0.5rem;'>None detected.</p>"
    pills = "".join(f'<span class="pill {pill_class}">{s}</span>' for s in skills)
    return f'<div class="pill-wrap">{pills}</div>'

def score_ring_svg(score):
    r = 70; circum = 2 * 3.14159 * r
    filled = circum * (score / 100); gap = circum - filled
    color = "#2affd0" if score >= 80 else ("#a88bff" if score >= 60 else "#ff5064")
    return f"""<div style="display:flex;flex-direction:column;align-items:center;padding:2rem 1rem;">
      <svg width="180" height="180" viewBox="0 0 180 180" style="filter:drop-shadow(0 0 18px rgba(42,255,208,0.3))">
        <circle cx="90" cy="90" r="{r}" fill="none" stroke="#1e1e2e" stroke-width="10"/>
        <circle cx="90" cy="90" r="{r}" fill="none" stroke="{color}" stroke-width="10"
          stroke-linecap="round" stroke-dasharray="{filled:.1f} {gap:.1f}" transform="rotate(-90 90 90)"/>
        <text x="90" y="84" text-anchor="middle" font-family="Syne,sans-serif"
          font-size="28" font-weight="800" fill="{color}">{score}</text>
        <text x="90" y="104" text-anchor="middle" font-family="DM Mono,monospace"
          font-size="10" fill="#55536a" letter-spacing="2">ATS SCORE</text>
      </svg>
      <p style="font-size:0.7rem;font-family:'DM Mono',monospace;letter-spacing:0.12em;
                text-transform:uppercase;color:#55536a;margin-top:0.3rem;">out of 100</p>
    </div>"""

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
      <div class="m-label">{label}</div><div class="m-value">{value}</div>
      {'<div class="m-sub">'+sub+'</div>' if sub else ''}
    </div>"""

def section_header(icon, title):
    return f"""<div class="section-header">
      <div class="sh-icon">{icon}</div><div class="sh-title">{title}</div>
    </div>"""

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
  <p>Upload your resume and paste a job description. Get a skills-based ATS score, missing keywords, and <strong>exact bullet-level fix suggestions.</strong></p>
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
# ANALYSIS
# ─────────────────────────────────────────────
if analyze:
    if resume_file is None or not jd_text.strip():
        st.warning("⚠️  Please upload a resume PDF and paste a job description.")
        st.stop()

    with st.spinner("Scanning resume, extracting skills, finding fix spots…"):

        # 1. Extract text
        resume_text = extract_text_from_pdf(resume_file)

        # 2. TF-IDF similarity
        similarity  = compute_similarity(resume_text, jd_text)
        match_pct   = round(similarity * 100, 1)

        # 3. Skills
        resume_skills   = extract_skills(resume_text)
        jd_skills       = extract_skills(jd_text)
        matching_skills = sorted(resume_skills & jd_skills)
        missing_skills  = sorted(jd_skills - resume_skills)
        extra_skills    = sorted(resume_skills - jd_skills)
        skill_pct       = round((len(matching_skills) / len(jd_skills)) * 100, 1) if jd_skills else 0.0
        final_ats       = ats_score(match_pct, skill_pct)

        # 4. Rank missing skills by JD frequency
        ranked_missing  = rank_missing_skills(set(missing_skills), jd_text)

        # 5. Extract bullets from resume
        bullets = extract_bullets(resume_text)

        # 6. Section completeness
        section_check = check_resume_sections(resume_text)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Scores ───────────────────────────────
    ring_col, metrics_col = st.columns([1, 2], gap="large")
    with ring_col:
        st.markdown(score_ring_svg(final_ats), unsafe_allow_html=True)
    with metrics_col:
        st.markdown(section_header("📊", "Score Breakdown"), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(metric_card("Match Score",  f"{match_pct}%",      "TF-IDF similarity"), unsafe_allow_html=True)
        with m2: st.markdown(metric_card("Skills Match", f"{skill_pct}%",      f"{len(matching_skills)} of {len(jd_skills)} skills"), unsafe_allow_html=True)
        with m3: st.markdown(metric_card("JD Skills",    str(len(jd_skills)),  "detected in JD"), unsafe_allow_html=True)

        if final_ats >= 80:
            st.markdown(verdict("🔥", "Excellent Match", " Your resume is strongly aligned. Apply with confidence.", "success"), unsafe_allow_html=True)
        elif final_ats >= 60:
            st.markdown(verdict("👍", "Good Match", " Solid profile — a few targeted additions can push you to the top tier.", "info"), unsafe_allow_html=True)
        else:
            st.markdown(verdict("⚠️", "Low Match", " Significant gaps detected. Tailor your resume before applying.", "error"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Skills Breakdown ─────────────────────
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

    # ── WHERE TO FIX — the key new section ───
    st.markdown(section_header("🔧", "Where to Fix It — Bullet-Level Suggestions"), unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#8b899e;font-size:0.85rem;margin-bottom:1.4rem;line-height:1.6;'>
    For each missing skill, the app found the most relevant line in your resume and suggests
    how to rewrite it. <span style='color:#2affd0;'>Highlighted text</span> shows exactly what to add.
    </p>""", unsafe_allow_html=True)

    if not ranked_missing:
        st.markdown("<p style='color:#55536a;'>No missing skills detected — great job!</p>", unsafe_allow_html=True)
    else:
        for skill, jd_count in ranked_missing:
            is_high = jd_count >= 2
            badge_cls = "high" if is_high else ""
            priority_label = "🔴 HIGH PRIORITY" if is_high else "🟡 SUGGESTED"

            best_bullet = find_best_bullet(skill, bullets)
            section_name = detect_section(best_bullet, resume_text) if best_bullet else "Skills"

            if best_bullet:
                rewritten = suggest_rewrite(best_bullet, skill)
                fix_html = f"""
                <div class="fix-card">
                  <span class="fix-skill-badge {badge_cls}">{priority_label} &nbsp;·&nbsp; {skill}</span>
                  <span class="fix-section-tag">📌 {section_name} Section</span>

                  <div class="fix-label old">📄 Original bullet in your resume</div>
                  <div class="fix-bullet-old">{best_bullet}</div>

                  <div class="fix-label new">✏️ Suggested rewrite (add highlighted part)</div>
                  <div class="fix-bullet-new">{rewritten}</div>
                </div>"""
            else:
                # No matching bullet found — suggest adding a new line
                skill_cap = skill.title() if len(skill) > 4 else skill.upper()
                fix_html = f"""
                <div class="fix-card">
                  <span class="fix-skill-badge {badge_cls}">{priority_label} &nbsp;·&nbsp; {skill}</span>
                  <span class="fix-section-tag">📌 Skills Section</span>

                  <div class="fix-label new">➕ No matching bullet found — add a new line</div>
                  <div class="fix-no-bullet">
                    Add <strong>{skill_cap}</strong> to your Skills section, or create a new project bullet:
                    <br><br>
                    <em>"Worked with {skill_cap} to [describe what you did or a relevant project outcome]."</em>
                  </div>
                </div>"""

            st.markdown(fix_html, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Resume Completeness Checklist ────────
    st.markdown(section_header("📋", "Resume Completeness Checklist"), unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#8b899e;font-size:0.85rem;margin-bottom:1rem;'>
    ATS systems and recruiters expect these sections. Missing ones can get your resume filtered out before a human reads it.
    </p>""", unsafe_allow_html=True)

    all_present = all(section_check.values())
    for section_name, found in section_check.items():
        icon  = "✅" if found else "❌"
        color = "#2affd0" if found else "#ff7088"
        note  = "" if found else f"<span style='color:#55536a;font-size:0.78rem;'>&nbsp;— Add this section to improve ATS visibility</span>"
        st.markdown(f"""
        <div class="check-row">
          <span class="check-icon">{icon}</span>
          <span style="color:{color};font-weight:600;">{section_name}</span>{note}
        </div>""", unsafe_allow_html=True)

    if all_present:
        st.markdown("<p style='color:#2affd0;font-size:0.85rem;margin-top:0.8rem;'>✨ All key sections detected — well structured resume!</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── General Recommendations ──────────────
    st.markdown(section_header("💡", "General Recommendations"), unsafe_allow_html=True)

    if match_pct < 55:
        st.markdown("""<div style="background:#13131f;border:1px solid #1e1e2e;border-left:3px solid #ffb347;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:0.8rem;font-size:0.88rem;color:#c0bdd4;line-height:1.6;">
            📝 <strong style='color:#e8e6f0;'>Mirror the JD's language.</strong>
            If the JD says <em>'data pipeline'</em>, use exactly that — not <em>'data workflow'</em>.
            ATS engines match phrases, not meaning.</div>""", unsafe_allow_html=True)

    if skill_pct < 40 and len(jd_skills) > 0:
        st.markdown("""<div style="background:#13131f;border:1px solid #1e1e2e;border-left:3px solid #ff5064;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:0.8rem;font-size:0.88rem;color:#c0bdd4;line-height:1.6;">
            📚 <strong style='color:#e8e6f0;'>Upskilling needed.</strong>
            Over 60% of required skills are missing. Focus on the top 2–3 high-priority
            missing skills with a quick project or certification before applying.</div>""", unsafe_allow_html=True)

    if final_ats >= 80:
        st.markdown("""<div style="background:#13131f;border:1px solid #1e1e2e;border-left:3px solid #2affd0;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:0.8rem;font-size:0.88rem;color:#c0bdd4;line-height:1.6;">
            ✨ <strong style='color:#e8e6f0;'>You're in great shape.</strong>
            Customise your summary line to echo the job title and key stack —
            it nudges human reviewers as much as the ATS.</div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
                border-top:1px solid #1e1e2e;color:#2a2a3e;font-size:0.72rem;
                font-family:'DM Mono',monospace;letter-spacing:0.1em;">
      BUILT WITH PYTHON · SKLEARN · PDFPLUMBER · STREAMLIT
    </div>""", unsafe_allow_html=True)

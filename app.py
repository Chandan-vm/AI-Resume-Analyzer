import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# SKILL DICTIONARY  (extend as needed)
# ─────────────────────────────────────────────

SKILL_DICTIONARY = {
    # ── Programming Languages ──────────────────
    "python", "r", "sql", "java", "scala", "julia",
    "c++", "c#", "javascript", "typescript", "bash", "shell",

    # ── Databases ─────────────────────────────
    "mysql", "postgresql", "sqlite", "mongodb", "cassandra",
    "redis", "bigquery", "snowflake", "redshift", "oracle",
    "sql server", "ms sql", "hive", "presto", "athena",

    # ── BI & Visualization ─────────────────────
    "power bi", "tableau", "looker", "metabase", "qlik",
    "matplotlib", "seaborn", "plotly", "ggplot", "d3.js",
    "excel", "google sheets", "data studio", "looker studio",

    # ── Data / Analytics ──────────────────────
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

    # ── Machine Learning / AI ─────────────────
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "supervised learning", "unsupervised learning",
    "reinforcement learning",
    "linear regression", "logistic regression",
    "decision tree", "random forest", "xgboost", "lightgbm",
    "gradient boosting", "neural network",
    "classification", "clustering", "dimensionality reduction",
    "pca", "k-means", "svm", "naive bayes",
    "model evaluation", "cross validation", "hyperparameter tuning",
    "mlops", "model deployment",

    # ── ML Frameworks & Libraries ─────────────
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
    "hugging face", "transformers", "spacy", "nltk",
    "pandas", "numpy", "scipy", "statsmodels",

    # ── Cloud Platforms ───────────────────────
    "aws", "azure", "gcp", "google cloud",
    "s3", "ec2", "lambda", "sagemaker",
    "azure ml", "databricks", "dataflow",

    # ── Big Data & Streaming ──────────────────
    "spark", "apache spark", "hadoop", "kafka",
    "airflow", "dbt", "luigi", "flink",

    # ── Version Control & Dev Tools ───────────
    "git", "github", "gitlab", "bitbucket",
    "docker", "kubernetes", "ci/cd", "jenkins",
    "jira", "confluence", "notion",

    # ── Product / Business Analysis ───────────
    "product analytics", "user research", "ux research",
    "requirement gathering", "business analysis",
    "stakeholder management", "roadmapping", "product roadmap",
    "agile", "scrum", "kanban", "sprint planning",
    "wireframing", "prototyping", "figma",
    "customer journey", "persona mapping",
    "market research", "competitive analysis",
    "okr", "north star metric",

    # ── Statistics ────────────────────────────
    "statistics", "probability", "bayesian",
    "time series", "forecasting", "arima",
    "distribution", "confidence interval", "p-value",

    # ── Soft skills (keep minimal) ────────────
    "communication", "presentation", "storytelling",
    "problem solving", "critical thinking",
}

# Pre-sort multi-word skills longest-first so "power bi" matches before "power"
SORTED_SKILLS = sorted(SKILL_DICTIONARY, key=len, reverse=True)


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace; keep hyphens and slashes for 'scikit-learn', 'ci/cd'."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-/#+]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> set:
    """
    Scan normalised text against the skill dictionary.
    Multi-word skills (e.g. 'power bi') are matched as substrings.
    Single words use whole-word boundaries to avoid false positives
    (e.g. 'r' matching inside 'report').
    """
    norm = normalise(text)
    found = set()

    for skill in SORTED_SKILLS:
        if " " in skill or "/" in skill or "-" in skill:
            # Multi-word / compound skill: substring match is fine
            if skill in norm:
                found.add(skill)
        else:
            # Single short word: require word boundaries
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, norm):
                found.add(skill)

    return found


def compute_similarity(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([normalise(text1), normalise(text2)])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def ats_score(match_pct: float, skill_pct: float) -> float:
    """
    Weighted formula:
      40 % → TF-IDF cosine similarity
      50 % → skill overlap
      10 % → base formatting bonus
    Capped at 100.
    """
    return min(round(match_pct * 0.40 + skill_pct * 0.50 + 10, 1), 100.0)


def score_label(score: float):
    if score >= 80:
        return "🔥 Excellent", "success"
    elif score >= 60:
        return "👍 Good", "info"
    else:
        return "⚠️ Low", "error"


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🎯 AI Resume Analyzer & ATS Matcher")
st.markdown(
    "Upload your resume and paste a job description to get a **skills-based ATS score** "
    "with actionable, domain-relevant suggestions."
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
with col2:
    jd_text = st.text_area("📋 Paste Job Description", height=220)

if st.button("🚀 Analyze Resume"):

    if resume_file is None or not jd_text.strip():
        st.warning("⚠️ Please upload a resume and paste a job description.")
        st.stop()

    with st.spinner("Analyzing your resume against the job description…"):

        resume_text = extract_text_from_pdf(resume_file)

        # ── Similarity (TF-IDF) ───────────────────
        similarity = compute_similarity(resume_text, jd_text)
        match_pct = round(similarity * 100, 1)

        # ── Skill extraction ──────────────────────
        resume_skills = extract_skills(resume_text)
        jd_skills     = extract_skills(jd_text)

        matching_skills = sorted(resume_skills & jd_skills)
        missing_skills  = sorted(jd_skills - resume_skills)
        extra_skills    = sorted(resume_skills - jd_skills)   # bonus: what you have beyond JD

        skill_pct = round(
            (len(matching_skills) / len(jd_skills)) * 100, 1
        ) if jd_skills else 0.0

        final_ats = ats_score(match_pct, skill_pct)
        label, level = score_label(final_ats)

    st.divider()

    # ── Metrics row ───────────────────────────────
    st.subheader("📊 Analysis Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ATS Score",      f"{final_ats} / 100")
    c2.metric("Match Score",    f"{match_pct}%")
    c3.metric("Skills Match",   f"{skill_pct}%")
    c4.metric("JD Skills Found", f"{len(matching_skills)} / {len(jd_skills)}")

    st.divider()

    # ── Skills breakdown ──────────────────────────
    st.subheader("🔍 Skills Breakdown")
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown("**✅ Matching Skills**")
        if matching_skills:
            for s in matching_skills:
                st.markdown(f"- `{s}`")
        else:
            st.caption("None found.")

    with k2:
        st.markdown("**❌ Missing Skills** *(add these to your resume)*")
        if missing_skills:
            for s in missing_skills:
                st.markdown(f"- `{s}`")
        else:
            st.success("No critical skills missing!")

    with k3:
        st.markdown("**➕ Extra Skills on Resume** *(not in this JD)*")
        if extra_skills:
            for s in extra_skills[:10]:
                st.markdown(f"- `{s}`")
        else:
            st.caption("None detected.")

    st.divider()

    # ── Suggestions ───────────────────────────────
    st.subheader("💡 Smart Recommendations")

    if level == "success":
        st.success(f"{label} — Your resume is strongly aligned with this role.")
    elif level == "info":
        st.info(f"{label} — Solid profile. A few targeted additions can push you into the top tier.")
    else:
        st.error(f"{label} — Significant gaps detected. Tailor your resume for this role.")

    if missing_skills:
        top_missing = missing_skills[:6]
        st.markdown(
            f"**🔧 Top skills to add / highlight:** `{'`, `'.join(top_missing)}`"
        )
        st.markdown(
            "Consider adding these in your Skills section, or weaving them "
            "into project bullet points if you have practical experience with them."
        )

    if match_pct < 55:
        st.markdown(
            "**📝 Rewrite tip:** Your resume language differs significantly from the JD. "
            "Mirror the JD's exact terminology (e.g. if the JD says *'data pipeline'*, use that phrase, not *'data workflow'*)."
        )

    if skill_pct < 40 and len(jd_skills) > 0:
        st.markdown(
            "**📚 Upskilling note:** More than 60 % of the required skills are missing. "
            "Focus on the top 2–3 missing tools with a quick project or certification before applying."
        )

    if not jd_skills:
        st.warning(
            "⚠️ No recognisable skills were found in the job description. "
            "The JD may be very generic — try pasting a more detailed version."
        )


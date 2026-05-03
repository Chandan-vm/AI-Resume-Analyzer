import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ---------- FUNCTIONS ----------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Remove useless + noise words
NOISE_WORDS = [
    "using", "based", "role", "team", "work", "working",
    "responsible", "required", "ability", "knowledge",
    "understanding", "experience", "familiarity",
    "skills", "good", "strong", "basic", "advanced",
    "building", "generating", "identify", "write",
    "candidate", "seeking", "ideal", "support",
    "large", "data", "process", "perform", "include",
    "develop", "ensure", "provide"
]

def get_keywords(text):
    words = text.split()

    keywords = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
        and word not in NOISE_WORDS
        and len(word) > 3
        and not word.endswith(("ing", "ed"))
    ]

    return set(keywords)

# ---------- UI ----------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🎯 AI Resume Analyzer & ATS Matcher")
st.markdown("Upload your resume and paste a job description to evaluate your ATS score and improve your resume.")

st.divider()

# ---------- INPUT ----------

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("📋 Paste Job Description", height=200)

# ---------- ANALYSIS ----------

if st.button("🚀 Analyze Resume"):

    if resume_file is not None and jd_text.strip() != "":

        with st.spinner("Analyzing your resume..."):

            resume_text = extract_text_from_pdf(resume_file)

            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_text)

            # TF-IDF Similarity
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_clean, jd_clean])

            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            match_score = round(similarity * 100, 2)

            # Keywords
            resume_words = get_keywords(resume_clean)
            jd_words = get_keywords(jd_clean)

            matching = sorted(list(resume_words.intersection(jd_words)))
            missing = sorted(list(jd_words.difference(resume_words)))

            skills_percent = round((len(matching) / len(jd_words)) * 100, 2) if jd_words else 0

            # ATS Score
            ats_score = round(
                match_score * 0.4 +
                skills_percent * 0.3 +
                20 + 10,
                2
            )

        st.divider()

        # ---------- RESULTS ----------

        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Match Score", f"{match_score}%")
        col2.metric("Skills Match", f"{skills_percent}%")
        col3.metric("ATS Score", f"{ats_score}")

        st.divider()

        # ---------- KEYWORDS ----------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matching Skills")
            st.write(matching[:12] if matching else "No matching skills found")

        with col2:
            st.subheader("❌ Missing Skills")
            st.write(missing[:12] if missing else "No major skills missing")

        st.divider()

        # ---------- SMART SUGGESTIONS ----------

        st.subheader("💡 Smart Recommendations")

        # 1. Skills Gap
        if len(missing) > 0:
            st.write("🔧 **Add missing skills/tools:**")
            st.write(", ".join(missing[:5]))

        # 2. Alignment Feedback
        if match_score < 60:
            st.write("📉 **Improve alignment:** Tailor your resume based on the job description. Focus on relevant experience and tools.")

        # 3. Skills Improvement
        if skills_percent < 50:
            st.write("🧠 **Strengthen your profile:** Add more relevant tools, technologies, or projects related to this role.")

        # 4. Resume Strength Feedback
        if ats_score > 80:
            st.success("🔥 Excellent! Your resume is highly aligned with this role.")

        elif 60 <= ats_score <= 80:
            st.info("👍 Good match! Improving a few skills can make your profile stronger.")

        else:
            st.error("⚠️ Low match. Consider revising your resume significantly for this job.")

    else:
        st.warning("⚠️ Please upload a resume and paste the job description.")


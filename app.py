import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
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

def get_keywords(text):
    return set(text.split())

# ---------- UI ----------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🎯 AI Resume Analyzer & ATS Matcher")

st.markdown("Upload your resume and paste a job description to analyze your ATS score.")

st.divider()

# ---------- INPUT SECTION ----------

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("📋 Paste Job Description", height=200)

# ---------- ANALYZE BUTTON ----------

if st.button("🚀 Analyze Resume"):

    if resume_file is not None and jd_text.strip() != "":

        with st.spinner("Analyzing... Please wait..."):

            resume_text = extract_text_from_pdf(resume_file)

            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_text)

            # TF-IDF
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_clean, jd_clean])

            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            match_score = round(similarity * 100, 2)

            # Keywords
            resume_words = get_keywords(resume_clean)
            jd_words = get_keywords(jd_clean)

            matching = resume_words.intersection(jd_words)
            missing = jd_words.difference(resume_words)

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
            st.subheader("✅ Matching Keywords")
            st.write(list(matching)[:15] if matching else "No matching keywords found")

        with col2:
            st.subheader("❌ Missing Keywords")
            st.write(list(missing)[:15] if missing else "No missing keywords")

        st.divider()

        # ---------- SUGGESTIONS ----------

        st.subheader("💡 Suggestions")

        if len(missing) > 0:
            st.write("👉 Add these keywords:", list(missing)[:5])

        if skills_percent < 50:
            st.write("👉 Improve skills alignment with the job description")

        if match_score < 60:
            st.write("👉 Resume does not strongly match the job description")

        if ats_score > 80:
            st.success("🔥 Excellent match! Your resume is well aligned.")

    else:
        st.warning("⚠️ Please upload a resume and paste the job description.")

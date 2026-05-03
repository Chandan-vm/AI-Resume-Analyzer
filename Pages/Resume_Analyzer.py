import streamlit as st
from utils import extract_text_from_pdf, analyze

st.title("📊 Resume Analyzer")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if resume_file and jd_text:
        resume_text = extract_text_from_pdf(resume_file)

        match, skills, ats, matching, missing = analyze(resume_text, jd_text)

        st.session_state["result"] = {
            "match": match,
            "skills": skills,
            "ats": ats,
            "matching": matching,
            "missing": missing
        }

        col1, col2, col3 = st.columns(3)
        col1.metric("Match Score", f"{match}%")
        col2.metric("Skills Match", f"{skills}%")
        col3.metric("ATS Score", f"{ats}")

    else:
        st.warning("Upload resume and paste JD")
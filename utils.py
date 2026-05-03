import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def analyze(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    match_score = round(similarity * 100, 2)

    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())

    matching = list(resume_words.intersection(jd_words))
    missing = list(jd_words.difference(resume_words))

    skills_percent = round((len(matching) / len(jd_words)) * 100, 2) if jd_words else 0

    ats_score = round(
        match_score * 0.4 +
        skills_percent * 0.3 +
        20 + 10,
        2
    )

    return match_score, skills_percent, ats_score, matching, missing
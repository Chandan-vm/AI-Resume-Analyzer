# 🎯 AI Resume Analyzer & ATS Matcher

## 📌 Overview

In today’s competitive job market, many candidates face repeated rejections not because of lack of skills, but because their resumes fail to align with job descriptions and Applicant Tracking Systems (ATS).

This project is an AI-powered Resume Analyzer that helps job seekers understand how well their resume matches a job description and provides actionable insights to improve their chances of getting shortlisted.

---

## ❗ Problem Statement

After facing multiple job rejections, I realized a key issue:

> Resumes often fail ATS screening due to poor keyword alignment and lack of relevance to the job description.

Most candidates:
- Do not know what keywords are missing
- Cannot quantify how well their resume matches a role
- Lack guidance on how to improve their resume

---

## 💡 Solution

To solve this, I built an intelligent system that:

- Compares resume and job description using NLP
- Calculates an ATS score based on multiple factors
- Identifies missing and matching keywords
- Provides personalized improvement suggestions

---

## 🚀 Key Features

- 📄 Upload Resume (PDF support)
- 📋 Paste Job Description
- 📊 ATS Score Calculation
- 🔑 Keyword Matching & Missing Keyword Detection
- 📈 Skills Match Percentage
- 💡 Personalized Resume Improvement Suggestions
- 🧭 Multi-page interactive Streamlit application

---

## 🧠 How It Works (Technical Explanation)

### 1. Text Extraction
- Resume text is extracted from PDF using `pdfplumber`

### 2. Text Preprocessing
- Lowercasing
- Removing special characters
- Tokenization

### 3. Feature Engineering
- TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical vectors

### 4. Similarity Calculation
- Cosine Similarity is used to measure how closely the resume matches the job description

### 5. Keyword Analysis
- Matching Keywords → Present in both resume & JD
- Missing Keywords → Present in JD but not in resume

### 6. ATS Scoring Logic

The final ATS score is calculated using weighted components:

- Keyword Match → 40%
- Skills Match → 30%
- Experience → 20%
- Education → 10%

---

## 📊 Output Example

The system provides:

- ATS Score (e.g., 72 → Good)
- Match Score (% similarity)
- Skills Match (%)
- Missing Keywords (e.g., Power BI, SQL)
- Suggestions (e.g., add tools, improve alignment)

---

## 🛠️ Tech Stack

- **Python**
- **NLP Techniques**
  - TF-IDF Vectorization
  - Cosine Similarity
- **Streamlit** (UI & deployment)
- **Scikit-learn**
- **PDFPlumber**

---

## 🧪 Why This Project Matters

This project is not just a technical implementation, but a real-world solution to a common problem faced by job seekers.

It demonstrates:

- Practical application of NLP in real-world scenarios
- Problem-solving mindset using personal experience
- Ability to build end-to-end data applications
- Understanding of ATS systems and hiring processes

---

## 🌍 Real-World Use Cases

- Job seekers optimizing resumes
- Students preparing for placements
- Career coaches guiding candidates
- Recruiters screening resumes efficiently

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer

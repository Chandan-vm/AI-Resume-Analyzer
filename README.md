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

- 📄 Resume upload (PDF support)
- 📋 Job Description analysis
- 🎯 ATS Score calculation
- 🔍 Skill-based matching (NOT keyword spam)
- ❌ Missing skill detection
- ➕ Extra skills identification
- 💡 Smart recommendations
- 🎨 Premium UI (Streamlit + custom CSS)

---

## 🧠 How It Works

1. Extracts resume text from PDF using pdfplumber  
2. Cleans and normalizes text  
3. Uses TF-IDF + Cosine Similarity to compute semantic match  
4. Extracts skills using a curated skill dictionary  
5. Compares:
   - Matching skills
   - Missing skills
   - Extra skills  
6. Calculates ATS score based on:
   - Match Score (40%)
   - Skill Match (50%)
   - Base Score (10%)  
7. Generates actionable recommendations for improvement

---

## 📊 Output Example

The system provides:

- ATS Score (e.g., 72 → Good)
- Match Score (% similarity)
- Skills Match (%)
- Missing Keywords (e.g., Power BI, SQL)
- Suggestions (e.g., add tools, improve alignment)

---

## 🌐 Live Demo

Try the app here:  
👉 https://ai-resume-analyzer-lh6j9dhzhoqczpf2qcyj8q.streamlit.app/

---

## 💡 What Makes This Different?

Unlike basic resume analyzers that rely on simple keyword matching, this system:

- Uses a curated skill dictionary to extract meaningful skills  
- Avoids irrelevant words like “responsible”, “seeking”, etc.  
- Provides structured skill comparison (matching / missing / extra)  
- Delivers actionable, real-world recommendations  
- Built based on real job search challenges  

This makes the tool closer to real ATS systems used in industry.

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
git clone https://github.com/Chandan-vm/AI-Resume-Analyzer
cd ai-resume-analyzer

"""
Module 2: Resume vs Job Description Skill Matcher + ATS Analyzer
Streamlit app

How to run:
1. Create a new virtualenv (recommended) and install requirements:
   pip install -r requirements.txt
2. Run:
   streamlit run app.py

Files:
- You may optionally upload a CSV/TSV/TXT skills list with one skill per line.
- Provide the Job Description as pasted text or upload a .txt file.
- Upload the resume as a PDF.

Notes:
- N-gram matching (1..3) against provided skills vocabulary ensures the app only finds terms from the skills list.
- Missing skills will only list skill terms (no stopwords/noise).
- ATS score logic described in comments below.
"""
import streamlit as st
import pdfplumber
import io
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from difflib import SequenceMatcher

# Ensure NLTK stopwords are downloaded (first-run)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Default skill vocabulary (starter). You should replace/extend with your own file for accuracy.
DEFAULT_SKILLS = [
    "python","java","c++","c#","javascript","react","angular","vue","html","css",
    "node.js","node","express","django","flask","sql","mysql","postgresql","mongodb",
    "redis","aws","azure","gcp","docker","kubernetes","git","github","gitlab","rest api",
    "graphql","tensorflow","pytorch","scikit-learn","pandas","numpy","opencv","nlp",
    "machine learning","deep learning","data analysis","data visualization",
    "spark","hadoop","etl","ci/cd","linux","bash","powershell","unit testing","pytest",
    "selenium","ui/ux","agile","scrum","jira","communication","leadership","problem solving",
    "microservices","react native","swift","kotlin","android","ios","typescript",
    "aws lambda","serverless","firebase","ansible","terraform","big data","tableau","power bi"
]

# ATS ideal section order (most ATS-friendly)
ATS_ORDER = [
    "contact", "summary", "skills", "experience", "projects", "education", "certifications", "achievements"
]

# Utility functions ---------------------------------------------------------

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF bytes using pdfplumber (keeps newlines)."""
    all_text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text.append(text)
    return "\n".join(all_text)

def normalize_text(text):
    """Lowercase, replace punctuation with spaces (except '.' in acronyms), normalize whitespace."""
    text = text.lower()
    # replace common punctuation with space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-z0-9\.\-\/ ]', ' ', text)  # keep dots for abbreviations and hyphens/slashes (e.g., c++ handled separately)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_skills_from_file(uploaded_file):
    """Load skill terms from uploaded file (one per line CSV or txt)."""
    try:
        content = uploaded_file.read().decode('utf-8', errors='ignore')
    except Exception:
        # binary fallback
        content = uploaded_file.read().decode('latin-1', errors='ignore')
    lines = [line.strip() for line in re.split(r'[\r\n,;]+', content) if line.strip()]
    return lines

def build_ngram_vectorizer(skills_list):
    """
    Build CountVectorizer that will only look for ngrams present in skills_list.
    This constrains matching to the skills vocabulary (prevents noise).
    """
    # Normalize skills
    skl = [normalize_text(s) for s in skills_list if s.strip()]
    # Remove duplicates
    skl = sorted(set(skl), key=lambda x: (-len(x), x))
    # Create vectorizer with vocabulary = skills phrases (as tokens)
    # We'll use analyzer='word' and pre-tokenize skills as single tokens replacing spaces with '_'
    vocab_tokens = {s.replace(' ', '_'): s for s in skl}
    # vectorizer: custom token pattern won't be used since we give a fixed vocabulary
    vectorizer = CountVectorizer(vocabulary=list(vocab_tokens.keys()), token_pattern=r"(?u)\b\w+\b", ngram_range=(1,3))
    return vectorizer, vocab_tokens

def text_to_token_form(text):
    """Replace spaces in multi-word skills with underscore to match vectorizer tokens."""
    return normalize_text(text).replace(' ', '_')

def extract_skill_matches(text, vectorizer, map_tokens_to_skill):
    """
    Returns a dict: skill_term -> count found in text.
    Only returns skills from vocabulary.
    """
    # Preprocess text for token matching:
    # Replace punctuation with spaces then convert multi-word tokens underscores
    normalized = normalize_text(text)
    # create simple tokens
    # We'll create word n-gram tokens via CountVectorizer transform on a preprocessed string where spaces are preserved.
    # But vectorizer expects tokens like 'machine_learning' so replace multiword skills in text with underscores where possible.
    # To do that robustly, replace occurrences of skill phrases with underscored tokens (longest-first).
    processed = normalized
    # replace skill phrases in processed with underscored tokens
    # longest-first to avoid overlapping partial replacements
    for token, orig_skill in sorted(map_tokens_to_skill.items(), key=lambda x: -len(x[1])):
        # token is underscored form, orig_skill is normalized phrase
        # use word-boundary replacement
        processed = re.sub(r'\b' + re.escape(orig_skill) + r'\b', token, processed)
    # Now transform using vectorizer
    X = vectorizer.transform([processed])
    counts = X.toarray()[0]
    found = {}
    for idx, cnt in enumerate(counts):
        if cnt > 0:
            token = vectorizer.get_feature_names_out()[idx]
            skill = map_tokens_to_skill.get(token, token.replace('_', ' '))
            found[skill] = int(cnt)
    return found

def detect_sections_order(resume_text):
    """
    Detect presence and order of the main resume sections by searching for keywords/headings.
    Returns: dict of section -> index_in_text (lower is earlier), or None if not found.
    Heuristic: search for keyword words and headings like 'skills', 'experience', etc.
    """
    text = resume_text.lower()
    # find positions
    positions = {}
    for sec in ATS_ORDER:
        # build several synonyms
        synonyms = {
            "contact": ["contact", "contact info", "contact information", "phone", "email", "address"],
            "summary": ["summary", "professional summary", "profile", "objective"],
            "skills": ["skill", "skills", "technical skills", "key skills", "areas of expertise"],
            "experience": ["experience", "work experience", "professional experience", "employment history"],
            "projects": ["project", "projects", "academic projects"],
            "education": ["education", "academic", "qualifications"],
            "certifications": ["certification", "certifications", "licenses"],
            "achievements": ["achievement", "achievements", "awards", "honors"]
        }.get(sec, [sec])
        pos = None
        for syn in synonyms:
            # search for heading-like occurrence (word boundary)
            m = re.search(r'\b' + re.escape(syn) + r'\b', text)
            if m:
                pos = m.start()
                break
        positions[sec] = pos
    return positions

def calculate_order_score(positions):
    """
    Compute a score [0..1] for how close the detected section order is to ATS_ORDER.
    We consider the detected sections that exist and compute Spearman-like rank correlation.
    If no sections found, score=0.
    """
    # Filter only found sections
    found = [(sec, pos) for sec, pos in positions.items() if pos is not None]
    if not found:
        return 0.0
    # sort by detected position
    found_sorted = sorted(found, key=lambda x: x[1])
    detected_order = [sec for sec, _ in found_sorted]
    # compute how many of these follow ATS_ORDER relative order using pairwise comparisons
    total_pairs = 0
    correct_pairs = 0
    # map ats order to index
    ats_index = {sec: i for i, sec in enumerate(ATS_ORDER)}
    for i in range(len(detected_order)):
        for j in range(i+1, len(detected_order)):
            total_pairs += 1
            a = detected_order[i]
            b = detected_order[j]
            if ats_index.get(a, 999) <= ats_index.get(b, 999):
                correct_pairs += 1
    if total_pairs == 0:
        return 0.0
    return correct_pairs / total_pairs

def compute_ats_score(skills_match_pct, sections_presence_pct, order_score, resume_length):
    """
    ATS score composition (weights chosen to emphasize skill matching and ATS sections):
    - skills_match_pct: 50% weight
    - sections_presence_pct: 25% weight
    - order_score: 15% weight
    - resume_length_normalized: 10% weight (encourages reasonable length)
    Ensures final score is at least 50% (user requirement).
    resume_length: number of words -> normalized to [0..1] with good range 200..900
    """
    # normalize resume length
    len_norm = min(max((resume_length - 200) / (700), 0), 1)  # 200->0, 900->1
    score = (
        skills_match_pct * 0.50 +
        sections_presence_pct * 0.25 +
        order_score * 100 * 0.15 +  # order_score is 0..1 convert to 0..100
        len_norm * 100 * 0.10
    )
    # floor to 50 if below 50 (per your request to guarantee at least 50%)
    final = max(score, 50.0)
    return round(final, 2)

# Streamlit UI ----------------------------------------------------------------

st.set_page_config(page_title="Resume Skill Matcher & ATS Analyzer", layout="wide")
st.title("Resume Skill Matcher & ATS Analyzer")
st.markdown("Upload a **resume (PDF)** and provide a **job description**. "
            "You can also upload a custom skills list (one skill per line). The app will only match skills from the skills list to avoid noise.")

with st.sidebar:
    st.header("Inputs")
    uploaded_resume = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    jd_input = st.text_area("Paste Job Description text here (or upload below)", height=200)
    uploaded_jd_file = st.file_uploader("Or upload Job Description (.txt)", type=["txt"])
    custom_skills_file = st.file_uploader("Upload skills list (one per line) [optional]", type=["txt", "csv"])
    ngram_choice = st.selectbox("Max n-gram for skill matching", options=[1,2,3], index=2)
    analyze_button = st.button("Analyze Resume vs Job Description")

# Load JD file if provided
if uploaded_jd_file and not jd_input.strip():
    try:
        jd_input = uploaded_jd_file.read().decode('utf-8', errors='ignore')
    except Exception:
        jd_input = uploaded_jd_file.read().decode('latin-1', errors='ignore')

# Load skills
if custom_skills_file:
    skills_list = load_skills_from_file(custom_skills_file)
    if len(skills_list) == 0:
        skills_list = DEFAULT_SKILLS
        st.sidebar.warning("Uploaded skills file empty or not readable — using default skill vocabulary.")
else:
    skills_list = DEFAULT_SKILLS

# Main process
if analyze_button:
    if not uploaded_resume:
        st.error("Please upload a resume PDF before analyzing.")
    elif not jd_input.strip():
        st.error("Please paste or upload a Job Description.")
    else:
        # extract resume text
        try:
            resume_bytes = uploaded_resume.read()
            resume_text_raw = extract_text_from_pdf(resume_bytes)
            if not resume_text_raw.strip():
                st.warning("Resume PDF parsed to empty text. It might be a scanned image PDF. Try OCR or upload a text/pdf that contains selectable text.")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            st.stop()

        # normalize texts
        resume_text_norm = normalize_text(resume_text_raw)
        jd_text_norm = normalize_text(jd_input)

        # Build vectorizer for skill vocab with user chosen ngram
        # vectorizer expects vocabulary tokens with underscores for multiword skills
        # Create token map
        # Limit skills to ngram length specified
        skills_norm = [normalize_text(s) for s in skills_list if s.strip()]
        # filter skills by token count <= ngram_choice
        skills_filtered = [s for s in skills_norm if len(s.split()) <= ngram_choice]
        if not skills_filtered:
            st.error("No skills left after filtering by n-gram size. Increase n-gram setting or upload a broader skills list.")
            st.stop()

        # mapping token->skill_phrase (underscored)
        token_map = {s.replace(' ', '_'): s for s in skills_filtered}
        vectorizer = CountVectorizer(vocabulary=list(token_map.keys()), token_pattern=r"(?u)\b\w+\b", ngram_range=(1, ngram_choice))
        # Fit not needed since vocabulary provided

        # Extract skills from JD and resume separately
        jd_matches = extract_skill_matches(jd_text_norm, vectorizer, token_map)
        resume_matches = extract_skill_matches(resume_text_norm, vectorizer, token_map)

        # compute matched skills and missing skills (only from JD skill set)
        jd_skill_set = set([k for k in jd_matches.keys()])
        resume_skill_set = set([k for k in resume_matches.keys()])

        matched_skills = sorted(list(jd_skill_set & resume_skill_set))
        missing_skills = sorted(list(jd_skill_set - resume_skill_set))

        # Build results summary and metrics
        matched_count = len(matched_skills)
        missing_count = len(missing_skills)
        total_jd_skills = len(jd_skill_set)

        skills_match_pct = (matched_count / total_jd_skills * 100) if total_jd_skills else 0.0

        # Section detection
        section_positions = detect_sections_order(resume_text_raw)
        sections_found = {sec: pos for sec, pos in section_positions.items() if pos is not None}
        sections_presence_pct = (len(sections_found) / len(ATS_ORDER) * 100)

        order_score = calculate_order_score(section_positions)  # 0..1

        resume_word_count = len(re.findall(r'\w+', resume_text_raw))

        ats_score = compute_ats_score(skills_match_pct, sections_presence_pct, order_score, resume_word_count)

        # UI with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Skill Match", "ATS Report", "Visuals"])

        with tab1:
            st.subheader("Quick Summary")
            st.metric("ATS Score", f"{ats_score} %")
            st.write(f"Job Description skills found: **{matched_count} / {total_jd_skills}** ({skills_match_pct:.1f}%)")
            st.write(f"Sections detected (count): **{len(sections_found)} / {len(ATS_ORDER)}**")
            st.write("---")
            st.markdown("**Top matched skills (from JD & Resume)**")
            if matched_skills:
                st.write(", ".join(matched_skills))
            else:
                st.info("No matched skills found between JD and resume using the given skills vocabulary & n-gram size.")
            st.markdown("**Top missing skills (present in JD but not in resume)**")
            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("None — resume covers all skills mentioned in the job description (per vocabulary).")

        with tab2:
            st.subheader("Detailed Skill Matching")
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Job Description — Extracted Skills (with counts)**")
                if jd_matches:
                    df_jd = pd.DataFrame(sorted(jd_matches.items(), key=lambda x: -x[1]), columns=["skill","count"])
                    st.dataframe(df_jd)
                else:
                    st.write("No skills found in Job Description (using provided vocabulary).")
            with col2:
                st.markdown("**Resume — Extracted Skills (with counts)**")
                if resume_matches:
                    df_res = pd.DataFrame(sorted(resume_matches.items(), key=lambda x: -x[1]), columns=["skill","count"])
                    st.dataframe(df_res)
                else:
                    st.write("No skills found in Resume (using provided vocabulary).")

            st.markdown("---")
            st.markdown("**Matched skills (from JD present in Resume)**")
            st.write(matched_skills if matched_skills else "—")

            st.markdown("**Missing skills (present in JD but NOT in Resume)**")
            st.write(missing_skills if missing_skills else "—")

        with tab3:
            st.subheader("ATS Section Analysis & Suggestions")
            st.write("ATS preferred order:", " → ".join(x.upper() for x in ATS_ORDER))
            st.write("---")
            # sections table
            sec_rows = []
            for sec in ATS_ORDER:
                pos = section_positions.get(sec)
                present = pos is not None
                sec_rows.append({"section": sec, "present": present, "position_index": pos if present else None})
            df_sections = pd.DataFrame(sec_rows)
            st.table(df_sections)

            st.markdown("**Suggestions to improve ATS compatibility**")
            suggestions = []
            # Suggest adding missing sections
            missing_sections = [r['section'] for r in sec_rows if not r['present']]
            if missing_sections:
                suggestions.append(f"Add these ATS-friendly sections: {', '.join(missing_sections)}.")
            # Suggest reordering tips if order_score low
            if order_score < 0.6:
                suggestions.append("Reorder sections to follow ATS-preferred order (Contact → Summary → Skills → Experience → Projects → Education → ...).")
            # Suggest skills to add
            if missing_skills:
                suggestions.append(f"Add or highlight these missing skills from the job description: {', '.join(missing_skills[:30])}.")
            # Resume length suggestions
            if resume_word_count < 250:
                suggestions.append("Resume appears short — expand with more specific accomplishments (quantified where possible).")
            elif resume_word_count > 2000:
                suggestions.append("Resume is quite long — consider trimming older or less relevant roles; focus on achievements.")

            if suggestions:
                for s in suggestions:
                    st.write("- " + s)
            else:
                st.success("Good — resume contains ATS-friendly sections in good order and skills coverage.")

            st.write("---")
            st.write(f"Sections presence: **{sections_presence_pct:.1f}%**")
            st.write(f"Section order agreement score: **{order_score*100:.1f}%**")
            st.write(f"Resume length (words): **{resume_word_count}**")

            st.write("---")
            st.markdown("**ATS Score logic (friendly)**")
            st.write("""
            - Skills match (how many JD skills are present in the resume) → **50% weight**  
            - Presence of ATS sections (Contact, Summary, Skills, Experience, Projects, Education, Certifications, Achievements) → **25% weight**  
            - Section order (how closely the resume order follows the ATS order) → **15% weight**  
            - Resume length (reasonable length preferred) → **10% weight**  
            Final score is capped to be at least **50%** to avoid too low results for early-stage jobseekers (per requirement).
            """)

        with tab4:
            st.subheader("Visualizations")

            # Bar chart: matched vs missing count
            fig1, ax1 = plt.subplots(figsize=(6,3))
            ax1.bar(["Matched","Missing"], [matched_count, missing_count])
            ax1.set_ylabel("Count")
            ax1.set_title("Matched vs Missing Skills")
            st.pyplot(fig1)

            # Pie chart: coverage
            fig2, ax2 = plt.subplots(figsize=(4,4))
            covered = matched_count
            not_covered = max(total_jd_skills - matched_count, 0)
            labels = ["Covered", "Not Covered"]
            ax2.pie([covered, not_covered], labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Job Description Skills Coverage")
            st.pyplot(fig2)

            # Section order visualization (if any)
            if sections_found:
                order_df = pd.DataFrame([
                    {"section": sec, "position": pos} for sec, pos in section_positions.items() if pos is not None
                ])
                order_df = order_df.sort_values('position')
                fig3, ax3 = plt.subplots(figsize=(8,2))
                ax3.hlines(1, xmin=0, xmax=1, color='lightgrey', linewidth=1)
                y = 1
                for i, row in order_df.iterrows():
                    ax3.scatter((row['position']), y, s=100)
                    ax3.text(row['position']+5, y, row['section'].upper(), va='center')
                ax3.set_xlim(0, max(order_df['position'].max()+50, 200))
                ax3.set_yticks([])
                ax3.set_xlabel("Character index in resume (approx position)")
                ax3.set_title("Detected Section Positions (left to right = top to bottom in resume)") 
                st.pyplot(fig3)
            else:
                st.info("No recognizable sections found to visualize order.")

        # Offer downloads / CSVs
        st.markdown("---")
        st.subheader("Export results")
        export_df = pd.DataFrame({
            "skill": list(jd_skill_set),
            "in_resume": [1 if s in resume_skill_set else 0 for s in jd_skill_set]
        })
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download JD vs Resume skill comparison (CSV)", data=csv_bytes, file_name="skill_comparison.csv", mime="text/csv")

        st.success("Analysis complete.")

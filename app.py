import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
df = pd.read_csv("data/medical_terms.csv")

# Create embeddings for terms
term_embeddings = model.encode(df["term"].tolist())

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

# ✅ NEW — Sentence-based chunking (better results)
def chunk_text(text):
    sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
    return sentences

# UI
st.title("AI Medical Report Analyzer")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_input = st.text_area("Or paste medical report text:")

if st.button("Analyze"):
    report_text = ""

    if uploaded_file is not None:
        report_text = extract_text_from_pdf(uploaded_file)
    elif user_input.strip():
        report_text = user_input.strip()

    if report_text:
        chunks = chunk_text(report_text)

        results = []

        for chunk in chunks:
            chunk_embedding = model.encode([chunk])
            similarity_scores = cosine_similarity(chunk_embedding, term_embeddings)[0]

            for i, score in enumerate(similarity_scores):
                results.append({
                    "chunk": chunk,
                    "term": df.iloc[i]["term"],
                    "explanation": df.iloc[i]["explanation"],
                    "score": score
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Sort by similarity
        results_df = results_df.sort_values(by="score", ascending=False)

        # Remove duplicate terms
        seen_terms = set()
        filtered_results = []

        for _, row in results_df.iterrows():
            if row["term"] not in seen_terms:
                filtered_results.append(row)
                seen_terms.add(row["term"])

            if len(filtered_results) == 5:
                break

        top_results = pd.DataFrame(filtered_results)

        # Display results
        st.subheader("Top Findings from Report")

        for _, row in top_results.iterrows():
            st.write(f"**Detected Issue:** {row['term']}")
            st.write(f"**Similarity Score:** {row['score']:.2f}")
            st.write(f"**Matched Report Chunk:** {row['chunk']}")
            st.write(f"**Explanation:** {row['explanation']}")
            st.write("---")

    else:
        st.warning("Please upload a PDF or paste some medical report text.")
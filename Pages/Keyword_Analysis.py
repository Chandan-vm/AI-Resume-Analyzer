import streamlit as st

st.title("🔑 Keyword Analysis")

if "result" in st.session_state:
    data = st.session_state["result"]

    st.subheader("✅ Matching Keywords")
    st.write(data["matching"][:20])

    st.subheader("❌ Missing Keywords")
    st.write(data["missing"][:20])
else:
    st.warning("Run analysis first")
import streamlit as st

st.title("💡 Suggestions")

if "result" in st.session_state:
    data = st.session_state["result"]

    if data["missing"]:
        st.write("👉 Add these keywords:", data["missing"][:5])

    if data["skills"] < 50:
        st.write("👉 Improve skills alignment")

    if data["match"] < 60:
        st.write("👉 Resume does not match JD well")

    if data["ats"] > 80:
        st.success("🔥 Excellent Resume Match!")

else:
    st.warning("Run analysis first")
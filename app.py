import streamlit as st
from config import validate_configuration, get_groq_model
from research_agent import run_research

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.22);
       border-radius: 18px; background: linear-gradient(135deg,#f8fafc,#eef2ff); margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2.25rem;}
.hero p {margin:.45rem 0 0; color:#64748b; font-size:1rem;}
.badge {display:inline-block; margin-top:.8rem; padding:.3rem .7rem; border-radius:999px;
        background:#dbeafe; color:#1d4ed8; font-size:.78rem; font-weight:600;}
.card {padding:1rem; border:1px solid rgba(128,128,128,.22); border-radius:14px; height:100%;}
.footer {color:#64748b; font-size:.78rem; margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🔬 AI Research Agent</h1>
<p>Evidence-grounded research with web discovery, source evaluation, cross-checking and cited synthesis.</p>
<div class="badge">Single-Agent • CrewAI • Groq • Web Research</div>
</div>
""", unsafe_allow_html=True)

error = validate_configuration()
if error:
    st.error(error)
    st.info("Add GROQ_API_KEY and GROQ_MODEL in Streamlit Cloud → App settings → Secrets.")
    st.stop()

with st.sidebar:
    st.header("Research Controls")
    depth = st.selectbox("Research depth", ["Standard", "Deep", "Comprehensive"], index=1)
    report_type = st.selectbox(
        "Report format",
        ["General Research Report", "Academic Research Report",
         "Business Research Report", "Market Research Report",
         "Technology Research Report"]
    )
    sources = st.slider("Target sources", 5, 15, 8)
    recency = st.selectbox(
        "Evidence recency",
        ["Any available date", "Last 5 years", "Last 3 years", "Last 12 months"]
    )
    st.divider()
    st.caption(f"LLM: {get_groq_model()}")
    st.caption("API credentials are read only from Streamlit Secrets.")

st.subheader("Research topic")
topic = st.text_area(
    "Enter a focused research question or topic",
    placeholder="Example: How is artificial intelligence affecting productivity in small businesses?",
    height=120,
    label_visibility="collapsed",
)

if st.button("🔬 Start Research", type="primary", use_container_width=True):
    if len(topic.strip()) < 10:
        st.warning("Please provide a more specific research topic.")
        st.stop()

    with st.status("Running research workflow…", expanded=True) as status:
        try:
            st.write("Planning research questions…")
            result = run_research(topic.strip(), depth, report_type, sources, recency)
            status.update(label="Research completed", state="complete", expanded=False)
            st.session_state["result"] = result
        except Exception as exc:
            status.update(label="Research failed", state="error", expanded=True)
            st.error("The research workflow could not be completed.")
            st.exception(exc)

result = st.session_state.get("result")
if result:
    st.divider()
    meta = result["metadata"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Results discovered", meta["results_discovered"])
    c2.metric("Sources used", meta["sources_used"])
    c3.metric("Source quality", meta["quality_summary"])
    c4.metric("Depth", meta["research_depth"])

    st.divider()
    st.markdown(result["report"])

    st.divider()
    st.subheader("Research sources")
    for i, src in enumerate(result["sources"], 1):
        with st.expander(f"{i}. {src.get('title','Untitled')}"):
            st.write(f"**Source type:** {src.get('source_type','Unknown')}")
            st.write(f"**Quality tier:** {src.get('quality_tier','Unknown')}")
            if src.get("published_date"):
                st.write(f"**Published:** {src['published_date']}")
            st.markdown(f"**URL:** {src['url']}")
            if src.get("evidence"):
                st.write("**Retrieved evidence:**")
                st.write(src["evidence"][:3000])

    st.download_button(
        "⬇️ Download report as Markdown",
        data=result["report"],
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.markdown(
    '<div class="footer">AI-generated research should be checked against original sources before high-stakes academic, legal, medical, financial or regulatory use.</div>',
    unsafe_allow_html=True,
)

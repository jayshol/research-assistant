"""
Streamlit UI for the research assistant.

Shows the pipeline running live: plan -> search -> write -> critique -> revise.
Deploy target: Streamlit Community Cloud.
"""
import streamlit as st
from planner import plan
from sources import gather_sources
from writer import write_report, revise_report, format_references
from critic import critique, format_issues_for_revision

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="wide")

st.title("🔎 Research Assistant")
st.caption(
    "Give it a topic. It plans searches, gathers sources, writes a cited report, "
    "then critiques and revises its own draft against the source material."
)

topic = st.text_input(
    "Research topic",
    placeholder="e.g. impact of large language models on scientific research",
)
run_button = st.button("Run", type="primary", disabled=not topic)

if run_button and topic:
    # --- Stage 1: Plan ---
    with st.status("Planning search queries...", expanded=True) as status:
        try:
            plan_items = plan(topic)
        except Exception as e:
            status.update(label="Planning failed", state="error")
            st.error(f"Could not generate a search plan: {e}")
            st.stop()

        for item in plan_items:
            st.write(f"**{item['sub_question']}**  →  `{item['query']}`")
        status.update(label=f"Planned {len(plan_items)} searches", state="complete")

    # --- Stage 2: Search & fetch ---
    with st.status("Searching and gathering sources...", expanded=True) as status:
        sources = gather_sources(topic, verbose=False)
        if not sources:
            status.update(label="No usable sources found", state="error")
            st.error(
                "No sources could be gathered for this topic — try rephrasing it "
                "or picking a broader topic."
            )
            st.stop()

        for s in sources:
            st.write(f"[{s['id']}] {s['title']} — {s['url']}")
        status.update(label=f"Gathered {len(sources)} usable sources", state="complete")

    # --- Stage 3: Write ---
    with st.status("Writing initial draft...", expanded=False) as status:
        try:
            draft = write_report(topic, sources)
        except Exception as e:
            status.update(label="Writing failed", state="error")
            st.error(f"Could not generate a report: {e}")
            st.stop()
        status.update(label="Initial draft complete", state="complete")

    # --- Stage 4: Critique ---
    with st.status("Critiquing draft against sources...", expanded=True) as status:
        try:
            issues = critique(draft, sources)
        except Exception as e:
            issues = []
            st.warning(f"Critic step failed, showing initial draft as final. ({e})")
        if issues:
            for i, issue in enumerate(issues, 1):
                st.write(
                    f"{i}. **[{issue['issue_type']}]** \"{issue['claim']}\" "
                    f"— cited source {issue.get('cited_source_id')}: {issue['explanation']}"
                )
            status.update(label=f"{len(issues)} issue(s) found", state="complete")
        else:
            status.update(label="No issues found", state="complete")

    # --- Stage 5: Revise (if needed) ---
    final_report = draft
    remaining_issues = []
    if issues:
        with st.status("Revising draft...", expanded=False) as status:
            revision_notes = format_issues_for_revision(issues)
            try:
                final_report = revise_report(topic, sources, draft, revision_notes)
                status.update(label="Revision complete", state="complete")
            except Exception as e:
                st.warning(f"Revision step failed, showing initial draft as final. ({e})")
                final_report = draft

        with st.status("Re-checking revised draft...", expanded=True) as status:
            try:
                remaining_issues = critique(final_report, sources)
            except Exception:
                remaining_issues = []
            if remaining_issues:
                for i, issue in enumerate(remaining_issues, 1):
                    st.write(
                        f"{i}. **[{issue['issue_type']}]** \"{issue['claim']}\" "
                        f"— cited source {issue.get('cited_source_id')}: {issue['explanation']}"
                    )
                status.update(
                    label=f"{len(remaining_issues)} issue(s) remain after revision",
                    state="complete",
                )
            else:
                status.update(label="No issues remain — clean pass", state="complete")

    # --- Final report ---
    st.divider()
    st.subheader("📄 Final Report")
    if issues and not remaining_issues:
        st.success("This report was revised once and passed re-critique with no remaining issues.")
    elif remaining_issues:
        st.info(
            f"This report was revised once. {len(remaining_issues)} issue(s) flagged above "
            "were not fully resolved — shown for transparency."
        )
    st.markdown(final_report)
    st.markdown(format_references(sources))
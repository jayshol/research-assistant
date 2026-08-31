"""
Streamlit UI for the research assistant.

Shows the pipeline running live: plan -> search -> write -> [critique -> revise]*.
Uses the bounded, convergence-tracked self-correction loop from revision_loop.py.
Deploy target: Streamlit Community Cloud.
"""
import streamlit as st
from planner import plan
from sources import gather_sources, get_last_debug_trace
from revision_loop import run_self_correcting_pipeline
from source_classifier import classify_sources
import usage_tracker
from analytics import assess_report, record_report_metrics, load_metrics_log, summarize_by_model

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="wide")

st.title("🔎 Research Assistant")
st.caption(
    "Give it a topic. It plans searches, gathers sources, writes a cited report, "
    "then iteratively critiques and revises its own draft against the source "
    "material until no issues remain (or a pass limit is hit)."
)

col_topic, col_passes = st.columns([4, 1])
with col_topic:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. impact of large language models on scientific research",
    )
with col_passes:
    max_passes = st.number_input(
        "Max critique passes", min_value=1, max_value=6, value=3, step=1,
        help="Upper bound on write -> critique -> revise cycles.",
    )

debug_mode = st.checkbox(
    "🐛 Debug mode — show generated queries and raw search results",
    value=False,
    help="Shows every planned search query alongside what Tavily actually returned for "
         "it, including results that were skipped during fetch (not just the sources "
         "that made it into the final report).",
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
        if debug_mode:
            with st.expander("🐛 Raw planner output (JSON)"):
                st.json(plan_items)
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
        sources = classify_sources(sources)

        if debug_mode:
            st.markdown("**🐛 Raw search results per query** (including skipped fetches)")
            for entry in get_last_debug_trace():
                st.write(f"**{entry['sub_question']}**  →  `{entry['query']}`")
                for r in entry["results"]:
                    if r["fetched"]:
                        st.write(f"✅ [{r['title']}]({r['url']}) — {r['chars']:,} chars extracted")
                    else:
                        st.write(f"⏭️ [{r['title']}]({r['url']}) — skipped (fetch failed or too short)")
                    if r.get("snippet"):
                        snippet = r["snippet"]
                        st.caption(snippet[:220] + ("…" if len(snippet) > 220 else ""))
        else:
            for s in sources:
                st.write(f"[{s['id']}] {s['title']} — {s['url']}  `{s['source_type_label']}`")
        status.update(label=f"Gathered {len(sources)} usable sources", state="complete")

    # --- Stage 3: Self-correcting write/critique/revise loop ---
    # We can't stream intermediate passes out of run_self_correcting_pipeline
    # without touching writer.py/critic.py, so run it as one step and then
    # unpack the full pass-by-pass trace from the result afterward.
    with st.status("Writing and self-correcting draft...", expanded=True) as status:
        usage_mark = usage_tracker.mark()
        try:
            result = run_self_correcting_pipeline(topic, sources, max_passes=max_passes)
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Could not generate a report: {e}")
            st.stop()

        for i, (count, issues) in enumerate(zip(result.issue_counts, result.issues_by_pass), 1):
            st.write(f"**Pass {i}:** {count} issue(s) found")
            for j, issue in enumerate(issues, 1):
                st.write(
                    f"&nbsp;&nbsp;{j}. **[{issue['issue_type']}]** \"{issue['claim']}\" "
                    f"— cited source {issue.get('cited_source_id')}: {issue['explanation']}"
                )

        label = (
            f"Converged after {result.num_revisions} revision(s)"
            if result.converged
            else f"Stopped at pass limit ({max_passes}) without full convergence"
        )
        status.update(label=label, state="complete")

    # --- Convergence summary ---
    st.divider()
    st.subheader("📉 Self-Correction Convergence")

    trace_str = " → ".join(str(n) for n in result.issue_counts)
    m1, m2, m3 = st.columns(3)
    m1.metric("Issue trace", trace_str)
    m2.metric("Revisions made", result.num_revisions)
    m3.metric("Converged to 0 issues", "Yes" if result.converged else "No")

    st.bar_chart(
        {"Issues found": result.issue_counts},
        x_label="Critique pass",
        y_label="Issue count",
        height=220,
    )

    if result.converged:
        st.success(result.summary_line())
    else:
        st.warning(result.summary_line())

    # --- Model performance (this run + accumulated history) ---
    metrics = assess_report(topic, sources, result, usage_mark=usage_mark)
    record_report_metrics(metrics)  # appended to metrics_log.jsonl -- accumulates across every run over time

    st.divider()
    st.subheader("⚙️ Model Performance")
    p1, p2, p3 = st.columns(3)
    p1.metric("Gemini calls (this run)", metrics["num_gemini_calls"])
    p2.metric("Total tokens (this run)", f"{metrics['total_tokens']:,}")
    p3.metric("Model", metrics["writer_model"])
    st.caption(
        "Logged to metrics_log.jsonl. This accumulates across every run (not just this "
        "session) -- useful later for comparing this model's performance against a "
        "different one if you swap the writer/critic model."
    )

    with st.expander("📊 Performance across all logged runs"):
        all_records = load_metrics_log()
        if len(all_records) <= 1:
            st.write("Not enough history yet -- run a few more topics to build up a comparison.")
        else:
            for model_key, stats in summarize_by_model(all_records).items():
                st.markdown(f"**{model_key}**")
                cols = st.columns(len(stats))
                for col, (k, v) in zip(cols, stats.items()):
                    col.metric(k.replace("_", " "), v)

    # --- Final report ---
    st.divider()
    st.subheader("📄 Final Report")
    st.markdown(result.final_report)
    st.markdown(result.references)

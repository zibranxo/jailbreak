"""Web-Based Admin Dashboard (Phase 7)."""

import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
import json

DB_PATH = Path("data/feedback.sqlite")

st.set_page_config(page_title="LLM Safety Admin Dashboard", layout="wide")

def get_db_connection():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)

st.title("🛡️ LLM Safety System - Admin Dashboard")
st.markdown("Monitor API usage, manage candidate patterns, and review active learning feedback.")

conn = get_db_connection()

if not conn:
    st.warning(f"Database not found at {DB_PATH}. Run the API and submit feedback to create it.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Metrics Overview", "🔍 Feedback Queue", "🧩 Candidate Patterns"])

with tab1:
    st.header("System Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    df_feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
    total_feedback = len(df_feedback)
    incorrect_preds = len(df_feedback[df_feedback['was_correct'] == 0])
    
    col1.metric("Total Feedback Items", total_feedback)
    col2.metric("Reported Misclassifications", incorrect_preds)
    col3.metric("Pending Retraining Samples", len(df_feedback[(df_feedback['was_correct'] == 0) & (df_feedback['processed'] == 0)]))
    
    if total_feedback > 0:
        st.subheader("Recent Feedback")
        st.dataframe(df_feedback.sort_values(by='timestamp', ascending=False).head(10))

with tab2:
    st.header("Review Misclassifications")
    st.markdown("Feedback where users indicated the system was incorrect.")
    
    df_incorrect = pd.read_sql_query("SELECT * FROM feedback WHERE was_correct = 0", conn)
    if df_incorrect.empty:
        st.info("No misclassifications reported!")
    else:
        st.dataframe(df_incorrect)
        if st.button("Mark All as Processed"):
            cur = conn.cursor()
            cur.execute("UPDATE feedback SET processed = 1 WHERE was_correct = 0")
            conn.commit()
            st.success("Successfully marked all feedback as processed. Retraining queue updated.")
            st.rerun()

with tab3:
    st.header("Pattern Auto-Discovery (Candidates)")
    st.markdown("Inputs that triggered embedding alerts but matched no explicit rules.")
    
    try:
        df_patterns = pd.read_sql_query("SELECT * FROM candidate_patterns", conn)
        if df_patterns.empty:
            st.info("No candidate patterns discovered yet.")
        else:
            pending = df_patterns[df_patterns['status'] == 'pending']
            st.metric("Pending Review", len(pending))
            st.dataframe(df_patterns)
            
            st.subheader("Actionable Patterns")
            for _, row in pending.iterrows():
                with st.expander(f"Pattern {row['id']}: {row['embedding_category']} (Sim: {row['similarity']:.2f})"):
                    st.text(row['text'])
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Approve & Add to Rules", key=f"app_{row['id']}"):
                            cur = conn.cursor()
                            cur.execute("UPDATE candidate_patterns SET status = 'approved' WHERE id = ?", (row['id'],))
                            conn.commit()
                            # Here we would also append to unsafe_patterns.jsonl
                            with open("data/patterns/unsafe_patterns.jsonl", "a") as f:
                                json.dump({"id": f"auto_{row['id']}", "pattern": row['text'], "category": row['embedding_category'], "type": "regex"}, f)
                                f.write("\n")
                            st.success("Approved and added to rules database.")
                            st.rerun()
                    with c2:
                        if st.button("Reject (False Alarm)", key=f"rej_{row['id']}"):
                            cur = conn.cursor()
                            cur.execute("UPDATE candidate_patterns SET status = 'rejected' WHERE id = ?", (row['id'],))
                            conn.commit()
                            st.warning("Rejected.")
                            st.rerun()
                            
    except Exception as e:
        st.error(f"Error loading candidate patterns: {e}")

st.sidebar.title("Configuration")
st.sidebar.markdown("P7.5 Dashboard Implementation")
st.sidebar.info("Connected to local SQLite database.")

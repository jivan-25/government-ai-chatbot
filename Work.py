# streamlit run govqa_app.py
import json, hashlib, time
from datetime import datetime
import pandas as pd
import psycopg2
import streamlit as st

# ==========================================
# Free AI Chatbot using Hugging Face
# No API key required
# ==========================================

# Install dependencies (run in terminal once):
# pip install transformers torch

from transformers import pipeline

def main():
    print("Loading chatbot model... (first run may take a minute)")
    
    # Load a small free model
    chatbot = pipeline("text-generation", model="distilgpt2")

    print("\n🤖 Chatbot is ready! Type 'quit' to exit.\n")

    while True:
        # Take user input
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye 👋")
            break

        # Generate a reply
        response = chatbot(
            user_input, 
            max_length=150,          # max words in response
            num_return_sequences=1,  # only 1 reply
            do_sample=True,          # randomness for creativity
            top_k=50,                # sampling top tokens
            temperature=0.7          # creativity level
        )[0]["generated_text"]

        # Clean response (remove repeating input)
        bot_reply = response[len(user_input):].strip()

        print("Bot:", bot_reply)
        print("-" * 50)  # separator for readability

if __name__ == "__main__":
    main()


st.set_page_config(page_title="GovQA", layout="wide")

# --- Config (replace with your secrets manager) ---
DB = dict(host="localhost", port=5432, dbname="gov", user="readonly", password="***")

# --- Policy / semantic registry (toy example) ---
SCOPES = {
    "finance": {
        "allowed_tables": ["gl_actuals", "gl_budget", "vendors", "calendar"],
        "metrics": {
            "budget_variance_ytd": {
                "sql": """
                SELECT cc, SUM(actual) AS actual, SUM(budget) AS budget,
                       SUM(actual) - SUM(budget) AS variance
                FROM (
                    SELECT a.cost_centre cc, a.amount actual, 0 budget, a.txn_date
                    FROM gl_actuals a
                    UNION ALL
                    SELECT b.cost_centre cc, 0 actual, b.amount budget, b.txn_date
                    FROM gl_budget b
                ) x
                WHERE txn_date >= date_trunc('year', current_date)
                GROUP BY cc
                """,
                "min_rows": 1
            }
        },
        "pii_columns": []
    },
    "hr": {
        "allowed_tables": ["employees", "leave", "org", "calendar"],
        "metrics": {
            "leave_rate_6m": {
                "sql": """
                WITH base AS (
                    SELECT e.team, l.emp_id, l.leave_days, l.leave_date
                    FROM leave l
                    JOIN employees e ON e.emp_id = l.emp_id
                    WHERE l.leave_date >= current_date - INTERVAL '6 months'
                ),
                agg AS (
                    SELECT team, COUNT(DISTINCT emp_id) AS ppl,
                           SUM(leave_days) AS days
                    FROM base GROUP BY team
                )
                SELECT team, days::float / NULLIF(ppl,0) AS leave_rate
                FROM agg
                """,
                "min_rows": 3,
                "min_cell_count": 10  # k-anonymity guard at row-level, enforced in code
            }
        },
        "pii_columns": ["dob", "ssn", "address"]
    },
    "ops": {
        "allowed_tables": ["po", "invoices", "vendors"],
        "metrics": {
            "payment_outliers": {
                "sql": """
                SELECT i.invoice_id, i.vendor_id, i.amount,
                       (i.amount - AVG(i.amount) OVER ())
                       / NULLIF(STDDEV(i.amount) OVER (),0) AS z
                FROM invoices i
                QUALIFY ABS(z) > 3
                """,
                "min_rows": 1
            }
        },
        "pii_columns": []
    }
}

def connect():
    return psycopg2.connect(**DB)

def compile_query(user_text, scope):
    """
    Tiny rules-based intent → metric mapping to avoid hallucinations.
    Extend with a small intent catalog instead of open-ended LLM answers.
    """
    t = user_text.lower()
    if scope == "finance" and ("budget" in t or "variance" in t):
        return "budget_variance_ytd"
    if scope == "hr" and ("leave" in t or "holiday" in t):
        return "leave_rate_6m"
    if scope == "ops" and ("outlier" in t or "anomaly" in t or "red flag" in t):
        return "payment_outliers"
    return None

def run_sql(sql):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)

def trust_score(df, policy, metric_name):
    score, reasons = 100, []
    # Row count
    min_rows = policy["metrics"][metric_name].get("min_rows", 1)
    if len(df) < min_rows:
        reasons.append("Low row count")
        score -= 40
    # PII scan (column names only in this toy example)
    pii = set(policy.get("pii_columns", []))
    if any(c.lower() in pii for c in df.columns):
        reasons.append("PII surfaced")
        score = min(score, 20)
    # Freshness check stub (replace with real table metadata)
    # Assume OK for demo
    return max(score, 0), reasons

def audit_record(user, scope, metric, sql, df, reasons, score):
    payload = {
        "ts": datetime.utcnow().isoformat(),
        "user": user,
        "scope": scope,
        "metric": metric,
        "sql": sql.strip(),
        "row_count": len(df),
        "columns": list(df.columns),
        "sample": df.head(10).to_dict(orient="records"),
        "trust_score": score,
        "reasons": reasons
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    payload["snapshot_hash"] = hashlib.sha256(blob).hexdigest()
    return payload

st.title("GovQA — Grounded, Auditable Conversational Analytics")

with st.sidebar:
    scope = st.selectbox("Scope", ["finance", "hr", "ops"])
    st.markdown("**Suggested questions**")
    if scope == "finance":
        st.write("- Am I going to meet my budget this year?\n- Show vendor payment outliers")
    elif scope == "hr":
        st.write("- What's happening with leave patterns in my team?")
    else:
        st.write("- Are there any red flags in our procurement data?")

q = st.text_input("Ask a question")
if st.button("Ask") and q:
    policy = SCOPES[scope]
    metric = compile_query(q, scope)

    if not metric:
        st.warning("I can only answer whitelisted questions for this scope. Try the suggestions in the sidebar.")
    else:
        sql = policy["metrics"][metric]["sql"]
        try:
            df = run_sql(sql)

            # Optional k-anonymity check for HR views
            if scope == "hr":
                mcc = policy["metrics"][metric].get("min_cell_count", 0)
                if "team" in df.columns and len(df) < mcc:
                    st.error("Result suppressed due to minimum cell size policy.")
                else:
                    score, reasons = trust_score(df, policy, metric)
                    audit = audit_record(st.session_state.get("user","analyst"), scope, metric, sql, df, reasons, score)

                    st.subheader(f"Answer • Trust {score}/100")
                    st.dataframe(df, use_container_width=True)

                    with st.expander("How I got this"):
                        st.code(sql, language="sql")
                        st.json(audit)

        except Exception as e:
            st.error(f"Query failed safely: {e}")

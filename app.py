from __future__ import annotations

import json
import html
import re
from pathlib import Path

import faiss
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer


INDEX_DIR = Path("vector_index")
DEFAULT_USER_CONTEXT = (
    "The user is an employee asking a workplace policy question. "
    "Infer the relevant policy area from the question and retrieve sections that directly answer it."
)

POLICY_AREAS = {
    "HR time off and leave": {
        "description": "Employee asking how to request time off, sick time, vacation, protected leave, or coverage planning.",
        "query_hints": "employee time off leave vacation sick time personal days request approval manager HR system coverage",
        "preferred_titles": {
            "Employee Leave and Time Off Policy": 0.48,
            "Remote Work and Device Policy": 0.06,
        },
        "preferred_section_terms": {
            "how to request time off": 0.3,
            "time off types": 0.14,
            "manager approval": 0.18,
            "sick time": 0.16,
            "extended leave": 0.14,
            "coverage planning": 0.12,
            "changing or canceling time off": 0.08,
            "unplanned absences": 0.12,
            "questions and exceptions": 0.1,
        },
        "avoid_titles": {
            "Procurement and Purchasing Policy": -0.2,
            "Vendor Onboarding and Risk Review Policy": -0.16,
        },
        "intent_terms": [
            "time off",
            "pto",
            "vacation",
            "sick",
            "leave",
            "absence",
            "holiday",
            "bereavement",
            "jury",
            "parental",
        ],
    },
    "Employee expense reports": {
        "description": "Employee asking about reimbursements, receipts, approvals, deadlines, or non-reimbursable expenses.",
        "query_hints": "employee reimbursement expense report receipt required documentation approval deadline non reimbursable",
        "preferred_titles": {
            "Employee Expense Reimbursement Policy": 0.45,
            "Remote Work and Device Policy": 0.12,
        },
        "preferred_section_terms": {
            "eligible expenses": 0.12,
            "home office equipment": 0.16,
            "professional certifications": 0.12,
            "travel and meals": 0.12,
            "required documentation": 0.22,
            "submission deadline": 0.2,
            "non-reimbursable expenses": 0.18,
            "exceptions": 0.1,
            "equipment and reimbursement": 0.14,
        },
        "avoid_titles": {
            "Procurement and Purchasing Policy": -0.16,
            "Vendor Onboarding and Risk Review Policy": -0.16,
        },
        "intent_terms": [
            "expense",
            "expenses",
            "expense report",
            "reimburse",
            "reimbursement",
            "receipt",
            "receipts",
            "home office",
            "monitor",
            "training",
            "certification",
            "mileage",
            "meal",
            "travel",
        ],
    },
    "Health benefits enrollment": {
        "description": "Employee asking about health benefits, medical, dental, vision, enrollment, dependents, coverage, or qualifying life events.",
        "query_hints": "employee health benefits enrollment medical dental vision coverage dependent benefits portal open enrollment qualifying life event",
        "preferred_titles": {
            "Health Benefits Enrollment Policy": 0.5,
            "Data Handling and Classification Policy": 0.06,
        },
        "preferred_section_terms": {
            "eligibility": 0.16,
            "new hire enrollment window": 0.28,
            "how to enroll": 0.3,
            "coverage start date": 0.18,
            "qualifying life events": 0.18,
            "dependent documentation": 0.16,
            "open enrollment": 0.16,
            "benefit questions and support": 0.14,
            "privacy": 0.08,
        },
        "avoid_titles": {
            "Responsible AI Usage Policy": -0.14,
            "Procurement and Purchasing Policy": -0.16,
            "Vendor Onboarding and Risk Review Policy": -0.16,
        },
        "intent_terms": [
            "benefit",
            "benefits",
            "health",
            "medical",
            "dental",
            "vision",
            "enroll",
            "enrollment",
            "coverage",
            "dependent",
            "dependents",
            "open enrollment",
            "qualifying life event",
            "life event",
        ],
    },
    "Responsible AI questions": {
        "description": "Employee or manager asking what data can be used with AI tools.",
        "query_hints": "approved AI tool restricted data human review privacy source citation prohibited use",
        "preferred_titles": {
            "Responsible AI Usage Policy": 0.42,
            "Data Handling and Classification Policy": 0.22,
        },
        "preferred_section_terms": {
            "restricted data": 0.2,
            "human review": 0.16,
            "approved ai tools": 0.14,
            "citation and source requirements": 0.12,
            "ai tool usage": 0.16,
        },
        "avoid_titles": {},
        "intent_terms": [
            "ai",
            "gemini",
            "chatgpt",
            "model",
            "prompt",
            "customer records",
            "customer data",
            "paste",
            "restricted data",
            "personal data",
        ],
    },
    "Vendor or software purchase": {
        "description": "Budget owner asking about vendor review, contracts, approvals, or software purchases.",
        "query_hints": "vendor software purchase procurement approval contract security privacy review quotes spend threshold",
        "preferred_titles": {
            "Procurement and Purchasing Policy": 0.35,
            "Vendor Onboarding and Risk Review Policy": 0.3,
            "Data Handling and Classification Policy": 0.08,
        },
        "preferred_section_terms": {
            "approval thresholds": 0.18,
            "software purchases": 0.22,
            "competitive quotes": 0.16,
            "security and privacy review": 0.18,
            "contract requirements": 0.14,
            "spend thresholds": 0.14,
        },
        "avoid_titles": {},
        "intent_terms": [
            "vendor",
            "software",
            "purchase",
            "purchasing",
            "procurement",
            "contract",
            "quote",
            "quotes",
            "budget",
            "invoice",
            "renewal",
        ],
    },
    "Customer support escalation": {
        "description": "Support agent asking when and how to escalate customer issues.",
        "query_hints": "support escalation customer severity security privacy unauthorized access incident ticket",
        "preferred_titles": {
            "Customer Support Escalation Policy": 0.42,
            "Security Incident Response Policy": 0.18,
        },
        "preferred_section_terms": {
            "severity levels": 0.16,
            "escalation timing": 0.18,
            "required escalation details": 0.22,
            "security and privacy concerns": 0.22,
            "customer communication": 0.14,
        },
        "avoid_titles": {},
        "intent_terms": [
            "support",
            "customer",
            "escalate",
            "escalation",
            "severity",
            "outage",
            "unauthorized access",
            "ticket",
            "incident",
            "fix date",
        ],
    },
    "General policy search": {
        "description": "Use broad retrieval across all policy documents.",
        "query_hints": "",
        "preferred_titles": {},
        "preferred_section_terms": {},
        "avoid_titles": {},
        "intent_terms": [],
    },
}


@st.cache_resource
def load_search_assets(index_dir: Path):
    index = faiss.read_index(str(index_dir / "policy.index"))
    records = json.loads((index_dir / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    model = None
    if manifest.get("embedder") == "sentence-transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(manifest["model"])
    return index, records, manifest, model


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def infer_policy_area(question: str, user_context: str) -> tuple[str, dict[str, int]]:
    text = f"{question} {user_context}".lower()
    scores: dict[str, int] = {}

    for area_name, area in POLICY_AREAS.items():
        if area_name == "General policy search":
            continue
        score = 0
        for term in area["intent_terms"]:
            if " " in term:
                if term in text:
                    score += 3
            elif re.search(rf"\b{re.escape(term)}\b", text):
                score += 1
        scores[area_name] = score

    best_area = max(scores, key=scores.get)
    if scores[best_area] == 0:
        return "General policy search", scores
    return best_area, scores


def build_search_query(question: str, policy_area: str, user_context: str) -> str:
    area = POLICY_AREAS[policy_area]
    parts = [question, user_context, area["description"], area["query_hints"]]
    return " ".join(part for part in parts if part)


def context_boost(record: dict, question: str, policy_area: str, user_context: str) -> float:
    area = POLICY_AREAS[policy_area]
    metadata = record["metadata"]
    title = metadata.get("title", "")
    section = metadata.get("section", "").lower()
    text = record["text"].lower()
    question_terms = tokenize(f"{question} {user_context}")

    boost = area["preferred_titles"].get(title, 0.0)
    boost += area["avoid_titles"].get(title, 0.0)

    for section_term, section_boost in area["preferred_section_terms"].items():
        if section_term in section:
            boost += section_boost

    metadata_text = f"{title} {section} {metadata.get('department', '')}".lower()
    matched_terms = sum(1 for term in question_terms if len(term) > 3 and term in f"{metadata_text} {text}")
    boost += min(matched_terms * 0.025, 0.15)

    if policy_area == "Employee expense reports":
        purchase_terms = {"vendor", "software", "contract", "procurement", "quote", "purchase"}
        if not question_terms.intersection(purchase_terms) and title == "Procurement and Purchasing Policy":
            boost -= 0.12

    if policy_area == "HR time off and leave" and "request" in question_terms and "time" in question_terms:
        if section == "3. how to request time off":
            boost += 0.16
        if section == "8. changing or canceling time off":
            boost -= 0.08

    return boost


def retrieve(question: str, policy_area: str, user_context: str, top_k: int = 5) -> list[dict]:
    index, records, manifest, model = load_search_assets(INDEX_DIR)
    search_query = build_search_query(question, policy_area, user_context)
    if manifest.get("embedder") == "sentence-transformer":
        query = model.encode([search_query], convert_to_numpy=True).astype("float32")
    else:
        vectorizer = HashingVectorizer(
            n_features=int(manifest.get("dimensions", 1024)),
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            stop_words="english",
        )
        query = vectorizer.transform([search_query]).toarray().astype("float32")
    faiss.normalize_L2(query)
    candidate_count = min(len(records), max(top_k * 5, 20))
    scores, ids = index.search(query, candidate_count)

    results = []
    for score, record_index in zip(scores[0], ids[0]):
        if record_index < 0:
            continue
        record = records[int(record_index)]
        boost = context_boost(record, question, policy_area, user_context)
        final_score = float(score) + boost
        results.append(
            {
                "score": float(score),
                "context_boost": boost,
                "final_score": final_score,
                **record,
            }
        )
    return sorted(results, key=lambda item: item["final_score"], reverse=True)[:top_k]


def risk_label(results: list[dict]) -> str:
    text = " ".join(result["text"].lower() for result in results)
    risks = []
    if any(term in text for term in ["payment", "expense", "reimbursement", "purchase", "invoice"]):
        risks.append("Finance")
    if any(term in text for term in ["legal", "contract", "liability", "regulator"]):
        risks.append("Legal")
    if any(term in text for term in ["security", "credential", "incident", "unauthorized"]):
        risks.append("Security")
    if any(term in text for term in ["personal data", "privacy", "customer data"]):
        risks.append("Privacy")
    if any(term in text for term in ["employee", "manager", "hr", "employment"]):
        risks.append("HR")
    return ", ".join(dict.fromkeys(risks)) if risks else "Low"


def citation_anchor(index: int) -> str:
    return f"source-{index}"


def clean_policy_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return " ".join(lines)


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def build_instructional_answer(results: list[dict]) -> str:
    primary = results[0]
    primary_meta = primary["metadata"]
    title = primary_meta.get("title", "")
    section = primary_meta.get("section", "")
    source = "[1](#source-1)"
    text = clean_policy_text(primary["text"])
    sentences = split_sentences(text)

    if title == "Employee Leave and Time Off Policy" and section == "3. How To Request Time Off":
        return (
            f"To request time off, submit the request through the HR system as early as practical {source}.\n\n"
            "Include:\n"
            "- The dates you want to take off\n"
            "- The type of time off\n"
            "- Your expected return date\n"
            "- Any coverage notes your manager needs\n\n"
            f"For planned vacation or personal days, submit the request at least 10 business days before your first day away when possible. Managers should respond within five business days {source}."
        )

    if title == "Health Benefits Enrollment Policy" and section == "4. How To Enroll":
        return (
            f"To enroll, go to the benefits portal and review the available medical, dental, and vision plans {source}.\n\n"
            "Before submitting, confirm your covered dependents, select coverage levels, designate beneficiaries when applicable, and submit your elections before the deadline.\n\n"
            f"After submitting, save or download the confirmation page. Benefits Administration may request dependent documentation {source}."
        )

    if title == "Employee Expense Reimbursement Policy" and section == "6. Required Documentation":
        return (
            f"For an expense report, include an itemized receipt, purchase date, vendor name, amount, business purpose, and approval evidence when pre-approval is required {source}.\n\n"
            f"A credit card statement alone usually is not enough unless Finance approves an exception {source}."
        )

    if not sentences:
        return f"Review the top matching policy section for guidance {source}."

    selected = sentences[: min(4, len(sentences))]
    return " ".join(selected) + f" {source}"


def build_demo_answer(question: str, results: list[dict], policy_area: str, user_context: str) -> str:
    if not results or results[0]["final_score"] < 0.25:
        return (
            "The uploaded policies do not provide enough information to answer this confidently. "
            "Route this to the policy owner or HR contact for review."
        )

    citations = [
        f"[{index}](#{citation_anchor(index)})"
        for index, _result in enumerate(results[:3], start=1)
    ]
    instructions = build_instructional_answer(results)
    return (
        f"{instructions}\n\n"
        f"Review the cited sections before taking action, especially if this affects pay, benefits, security, or protected leave. {' '.join(citations)}\n\n"
        f"Risk flag: {risk_label(results)}\n\n"
        "Sources: " + " ".join(citations)
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

        :root {
            --paper: #f3eadc;
            --panel: #fff8ed;
            --ink: #34291f;
            --muted: #786956;
            --line: #dac7ad;
            --accent: #5b6f3a;
            --accent-ink: #fffdf6;
            --source: #e7ead6;
            --lichen: #7f8a52;
            --walnut: #5a4030;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(127, 138, 82, 0.14), transparent 34rem),
                radial-gradient(circle at bottom right, rgba(91, 111, 58, 0.11), transparent 32rem),
                linear-gradient(180deg, var(--paper), #ead9c4 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 900px;
            padding-top: 3.2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background: transparent;
        }

        .hero {
            text-align: center;
            margin: 1.4rem auto 2rem;
        }

        .hero h1 {
            color: var(--ink);
            font-family: "Source Serif 4", Georgia, serif;
            font-size: clamp(2.8rem, 8vw, 5.2rem);
            font-weight: 600;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.02rem;
            margin: 0 auto;
            max-width: 38rem;
        }

        [data-testid="stTextInput"] input {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 999px;
            box-shadow: 0 14px 38px rgba(76, 62, 44, 0.12);
            color: var(--ink);
            font-size: 1.05rem;
            min-height: 4rem;
            padding: 0 1.5rem;
        }

        [data-testid="stTextInput"] input:focus {
            border-color: rgba(91, 111, 58, 0.72);
            box-shadow: 0 0 0 0.18rem rgba(91, 111, 58, 0.18), 0 14px 38px rgba(76, 62, 44, 0.12);
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid #b99b76;
            background: #fbf1e3;
            color: var(--ink);
            min-height: 2.75rem;
            padding: 0 1.25rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: var(--walnut);
            background: #ead6bd;
            color: var(--ink);
        }

        .stButton > button[kind="primary"],
        .stForm .stButton > button {
            border-color: var(--accent);
            background: var(--accent);
            color: var(--accent-ink);
        }

        .stButton > button[kind="primary"]:hover,
        .stForm .stButton > button:hover {
            border-color: #45552c;
            background: #45552c;
            color: #ffffff;
        }

        div[data-testid="stTabs"] button {
            color: var(--muted);
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ink);
        }

        .answer-card, .source-card {
            background: rgba(255, 253, 248, 0.88);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 12px 32px rgba(82, 59, 39, 0.1);
            margin-top: 1rem;
            padding: 1.15rem 1.25rem;
        }

        .answer-card h3, .source-card h4 {
            color: var(--ink);
            font-weight: 650;
            letter-spacing: 0;
            margin: 0 0 0.7rem;
        }

        .source-card {
            background: var(--source);
            border-color: #c8cda9;
        }

        .source-meta {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.65rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            background: #e8e2c8;
            border: 1px solid #c9c08f;
            border-radius: 999px;
            color: #4f5d31;
            font-size: 0.82rem;
            font-weight: 650;
            margin: 0.1rem 0 0.85rem;
            padding: 0.28rem 0.7rem;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.65rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(index: int, match: dict) -> None:
    meta = match["metadata"]
    title = html.escape(meta["title"])
    section = html.escape(meta["section"])
    source_file = html.escape(meta.get("source_file", "unknown source"))
    text = html.escape(match["text"])
    st.markdown(
        f"""
        <section id="{citation_anchor(index)}" class="source-card">
            <h4>[{index}] {title}</h4>
            <div class="source-meta">Section {section} · {source_file}</div>
            <div>{text}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def run_search(question: str, top_k: int) -> None:
    policy_area, intent_scores = infer_policy_area(question, DEFAULT_USER_CONTEXT)
    matches = retrieve(
        question,
        policy_area=policy_area,
        user_context=DEFAULT_USER_CONTEXT,
        top_k=top_k,
    )
    st.session_state["search_result"] = {
        "question": question,
        "policy_area": policy_area,
        "intent_scores": intent_scores,
        "matches": matches,
    }


st.set_page_config(page_title="Your HR Assistant", page_icon="HR", layout="centered")
inject_styles()

st.markdown(
    """
    <div class="hero">
        <h1>Your HR Assistant</h1>
        <p>Ask a workplace policy question and get a grounded answer with numbered sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not (INDEX_DIR / "policy.index").exists():
    st.warning("No local index found. Run `python scripts/ingest_faiss.py` first.")
    st.stop()

examples = [
    "How do I request time off?",
    "How do I enroll in health benefits?",
    "Can I get reimbursed for a home office monitor?",
    "Can I paste customer data into an AI tool?",
]

if "question" not in st.session_state:
    st.session_state["question"] = examples[0]

ask_tab, hood_tab = st.tabs(["Ask", "Under the hood"])

with ask_tab:
    with st.form("ask_form"):
        question = st.text_input(
            "Ask a policy question",
            key="question",
            label_visibility="collapsed",
            placeholder="Ask a policy question",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    cols = st.columns(len(examples))
    for index, example in enumerate(examples):
        if cols[index].button(example, key=f"example_{index}"):
            st.session_state["question"] = example
            run_search(example, top_k=5)

    if submitted and question.strip():
        run_search(question.strip(), top_k=5)

    result = st.session_state.get("search_result")
    if result:
        matches = result["matches"]
        policy_area = result["policy_area"]
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.markdown("### Answer")
        st.markdown(f'<div class="pill">{html.escape(policy_area)}</div>', unsafe_allow_html=True)
        st.markdown(
            build_demo_answer(
                result["question"],
                matches,
                policy_area,
                DEFAULT_USER_CONTEXT,
            )
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Sources")
        for index, match in enumerate(matches[:3], start=1):
            render_source_card(index, match)
    else:
        st.markdown(
            '<p class="small-note">Try asking about time off, benefits, expenses, devices, AI usage, vendors, or support escalation.</p>',
            unsafe_allow_html=True,
        )

with hood_tab:
    st.subheader("Under the hood")
    top_k = st.slider("Retrieved sections", min_value=3, max_value=8, value=5)
    if st.button("Rerun retrieval with this setting"):
        current_question = st.session_state.get("question", "").strip()
        if current_question:
            run_search(current_question, top_k=top_k)

    result = st.session_state.get("search_result")
    if result:
        st.caption(f"Inferred policy area: {result['policy_area']}")
        st.caption(
            "Intent scores: "
            + ", ".join(
                f"{area}: {score}" for area, score in result["intent_scores"].items()
            )
        )
        for index, match in enumerate(result["matches"], start=1):
            meta = match["metadata"]
            label = (
                f"[{index}] {match['final_score']:.3f} "
                f"(vector {match['score']:.3f}, context {match['context_boost']:+.3f}) "
                f"- {meta['title']} / {meta['section']}"
            )
            with st.expander(label):
                st.markdown(
                    f"**[{index}] {meta['title']}**  \n"
                    f"Section {meta['section']}  \n"
                    f"`{meta.get('source_file', 'unknown source')}`"
                )
                st.write(match["text"])
    else:
        st.info("Run a search to inspect inferred intent, scores, and retrieved sections.")

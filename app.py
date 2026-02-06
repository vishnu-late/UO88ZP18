"""
AI Customer Support Assistant - College FAQ Chatbot
==================================================
A Flask-based chatbot that answers institutional FAQs using TF-IDF and cosine similarity.
"""

import csv
import os
from datetime import datetime, timezone

# Optional: load environment variables from a local ".env" file (beginner-friendly).
# This lets you store GEMINI_API_KEY in a file instead of typing it every time.
# The ".env" file should NOT be committed to Git (we add it to .gitignore).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, we just skip .env loading.
    pass

from flask import Flask, render_template, request, jsonify, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------------------------------------------------------
# STEP 1: Initialize Flask App
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Allow microphone use + allow embedding on jspmntc.edu.in (iframe widget).
@app.after_request
def add_permissions_policy_headers(response):
    response.headers.setdefault("Permissions-Policy", "microphone=(self)")
    # Allow the main college site to embed this chatbot in an iframe.
    # NOTE: if you deploy under a different domain/subdomain, update these.
    response.headers.setdefault(
        "Content-Security-Policy",
        "frame-ancestors 'self' https://jspmntc.edu.in https://www.jspmntc.edu.in;",
    )
    return response

# Confidence threshold - below this we ask user to contact office
# Lower = answers more often; higher = asks user to contact office more often.
CONFIDENCE_THRESHOLD = 0.22

# Paths to data files
FAQ_FILE = os.path.join(os.path.dirname(__file__), 'faq.csv')
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), 'feedback.csv')

# New (richer) feedback columns. We keep the same file name (`feedback.csv`) and
# auto-migrate older files (with fewer columns) at runtime.
FEEDBACK_COLUMNS = [
    "timestamp_utc",
    "question",
    "answer",
    "feedback",
    "confidence",
    "matched_faq",
    "comment",
]

# Optional: Gemini API key (Google AI Studio).
# Set it in your terminal before running:
#   PowerShell:  $env:GEMINI_API_KEY="YOUR_KEY"
#   CMD:         set GEMINI_API_KEY=YOUR_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


# -----------------------------------------------------------------------------
# STEP 2: Load FAQ Data and Build Similarity Model
# -----------------------------------------------------------------------------

def load_faq_data():
    """
    Load questions and answers from faq.csv
    Returns: (questions list, answers list)
    """
    questions = []
    answers = []
    
    with open(FAQ_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['question'].strip().lower())
            answers.append(row['answer'].strip())
    
    return questions, answers


def build_faq_model():
    """
    Build TF-IDF vectorizer and fit it on FAQ questions.
    TF-IDF converts text to numerical vectors based on term importance.
    """
    questions, answers = load_faq_data()
    
    # Create TF-IDF vectorizer - converts text to numerical vectors
    # min_df=1 means include terms that appear in at least 1 document
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',  # Remove common words like 'the', 'is', etc.
        ngram_range=(1, 2)     # Use unigrams and bigrams for better matching
    )
    
    # Fit on FAQ questions and transform to vectors
    question_vectors = vectorizer.fit_transform(questions)
    
    return vectorizer, question_vectors, questions, answers


# Load data and build model at startup
vectorizer, question_vectors, faq_questions, faq_answers = build_faq_model()
# Track FAQ file modified time so we can reload when faq.csv changes
FAQ_LAST_MTIME = os.path.getmtime(FAQ_FILE) if os.path.exists(FAQ_FILE) else 0.0


def ensure_faq_model_uptodate() -> None:
    """
    Reload `faq.csv` + rebuild TF-IDF model if the file changed on disk.

    This fixes the common beginner issue where you edit `faq.csv` but the running
    Flask server still uses the old in-memory vectors until a restart.
    """
    global vectorizer, question_vectors, faq_questions, faq_answers, FAQ_LAST_MTIME

    try:
        mtime = os.path.getmtime(FAQ_FILE)
    except OSError:
        return

    if mtime != FAQ_LAST_MTIME:
        vectorizer, question_vectors, faq_questions, faq_answers = build_faq_model()
        FAQ_LAST_MTIME = mtime

def try_gemini_answer(user_question: str) -> str | None:
    """
    Optional Gemini fallback.

    - Only runs if GEMINI_API_KEY is set.
    - Uses the FAQ content as grounding text.
    - Returns a string answer, or None if Gemini is unavailable/fails.
    """
    if not GEMINI_API_KEY:
        return None

    # Import only when needed so the app still runs without Gemini installed.
    try:
        import google.generativeai as genai
    except Exception:
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Provide the FAQ list as context and instruct the model to answer like an FAQ bot.
        faq_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(faq_questions, faq_answers)])
        prompt = (
            "You are a college office FAQ assistant. Answer ONLY using the FAQ context below. "
            "If the answer is not present in the context, reply exactly:\n"
            "\"I am not sure about this. Please contact the office.\"\n\n"
            f"FAQ context:\n{faq_context}\n\n"
            f"User question: {user_question}\n"
            "Answer:"
        )

        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", None) or "").strip()
        return text or None
    except Exception:
        return None

def get_answer(user_question):
    """
    Find the best matching FAQ answer using cosine similarity.
    
    Args:
        user_question: The question asked by the user
        
    Returns:
        tuple: (answer_string, confidence_score)
    """
    # Handle empty input
    if not user_question or not user_question.strip():
        return "Please enter a question.", 0.0

    # --- Small talk handling (so "hi", "ok", "thanks" don't feel broken) ---
    normalized = user_question.strip().lower()
    ensure_faq_model_uptodate()
    if normalized in {"hi", "hello", "hey", "hii", "hiii"}:
        return (
            "Hi! Ask me something like:\n"
            "- What are the office timings?\n"
            "- How much is the tuition fee?\n"
            "- When are the semester exams held?",
            1.0,
        )
    if normalized in {"thanks", "thank you", "thx", "ty"}:
        return ("You're welcome! If you have another question, ask away.", 1.0)
    if normalized in {"bye", "goodbye", "see you"}:
        return ("Goodbye! Have a great day.", 1.0)
    if normalized in {"ok", "okay", "kk", "k"}:
        return ("Sure — ask your question and I’ll find the closest FAQ answer.", 1.0)
    
    # Transform user question to same vector space
    user_vector = vectorizer.transform([normalized])
    
    # Compute cosine similarity between user question and all FAQs
    # Returns values between 0 (no similarity) and 1 (identical)
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    
    # Get index of best matching FAQ
    best_match_idx = np.argmax(similarities)
    confidence = float(similarities[best_match_idx])
    
    # If confidence is low, try Gemini fallback (if configured), otherwise suggest contacting office.
    if confidence < CONFIDENCE_THRESHOLD:
        gemini_text = try_gemini_answer(user_question)
        if gemini_text:
            return gemini_text, confidence
        return ("I am not sure about this. Please contact the office.", confidence)
    
    return faq_answers[best_match_idx], confidence


def _migrate_feedback_csv_if_needed() -> None:
    """
    Ensure `feedback.csv` has the latest header (FEEDBACK_COLUMNS).

    If the file doesn't exist: create it with the latest header.
    If it exists with an older header: rewrite into the new header, preserving old data.
    """
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
            writer.writeheader()
        return

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except Exception:
        header = None

    if header == FEEDBACK_COLUMNS:
        return

    # Read existing rows using best-effort mapping.
    rows: list[dict] = []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8", newline="") as f:
            dict_reader = csv.DictReader(f)
            for r in dict_reader:
                rows.append(r)
    except Exception:
        rows = []

    # Rewrite file with new header + migrated rows.
    with open(FEEDBACK_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "timestamp_utc": r.get("timestamp_utc", "") or "",
                    "question": r.get("question", "") or "",
                    "answer": r.get("answer", "") or "",
                    "feedback": r.get("feedback", "") or "",
                    "confidence": r.get("confidence", "") or "",
                    "matched_faq": r.get("matched_faq", "") or "",
                    "comment": r.get("comment", "") or "",
                }
            )


def save_feedback(question, answer, feedback_type, confidence=None, matched_faq="", comment=""):
    """
    Append feedback to feedback.csv
    feedback_type: 'helpful' or 'not_helpful'
    """
    _migrate_feedback_csv_if_needed()

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conf_val = ""
    if isinstance(confidence, (int, float)):
        conf_val = str(round(float(confidence), 4))

    with open(FEEDBACK_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        writer.writerow(
            {
                "timestamp_utc": ts,
                "question": question or "",
                "answer": answer or "",
                "feedback": feedback_type or "",
                "confidence": conf_val,
                "matched_faq": matched_faq or "",
                "comment": comment or "",
            }
        )


# -----------------------------------------------------------------------------
# STEP 3: Define Routes
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')


@app.route("/widget.js")
def widget_js():
    """
    Embeddable widget script.

    Usage on jspmntc.edu.in (add before </body>):
      <script src="https://YOUR-CHATBOT-DOMAIN/widget.js" defer></script>
    """
    js = r"""
(function () {
  var WIDGET_ID = "jspm-faq-widget";
  if (document.getElementById(WIDGET_ID)) return;

  var CHAT_ORIGIN = window.__CHATBOT_ORIGIN__ || ""; // optional override
  var CHAT_URL = (CHAT_ORIGIN ? CHAT_ORIGIN : "") + "/";

  var root = document.createElement("div");
  root.id = WIDGET_ID;
  root.style.position = "fixed";
  root.style.right = "18px";
  root.style.bottom = "18px";
  root.style.zIndex = "2147483647";
  root.style.fontFamily = "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif";

  var btn = document.createElement("button");
  btn.type = "button";
  btn.setAttribute("aria-label", "Open chat assistant");
  btn.textContent = "Chat";
  btn.style.width = "56px";
  btn.style.height = "56px";
  btn.style.borderRadius = "16px";
  btn.style.border = "0";
  btn.style.cursor = "pointer";
  btn.style.background = "#2563eb";
  btn.style.color = "white";
  btn.style.fontWeight = "700";
  btn.style.boxShadow = "0 12px 30px rgba(2,6,23,.24)";
  btn.style.transition = "transform .12s ease";
  btn.onmouseenter = function () { btn.style.transform = "translateY(-1px)"; };
  btn.onmouseleave = function () { btn.style.transform = "translateY(0px)"; };

  var panel = document.createElement("div");
  panel.style.position = "absolute";
  panel.style.right = "0";
  panel.style.bottom = "68px";
  panel.style.width = "min(420px, calc(100vw - 36px))";
  panel.style.height = "min(640px, calc(100vh - 120px))";
  panel.style.borderRadius = "18px";
  panel.style.overflow = "hidden";
  panel.style.boxShadow = "0 20px 60px rgba(2,6,23,.28)";
  panel.style.border = "1px solid rgba(15,23,42,.12)";
  panel.style.background = "white";
  panel.style.display = "none";

  var iframe = document.createElement("iframe");
  iframe.src = CHAT_URL;
  iframe.title = "College FAQ Assistant";
  iframe.allow = "microphone; clipboard-write";
  iframe.referrerPolicy = "no-referrer-when-downgrade";
  iframe.style.width = "100%";
  iframe.style.height = "100%";
  iframe.style.border = "0";

  var close = document.createElement("button");
  close.type = "button";
  close.textContent = "×";
  close.setAttribute("aria-label", "Close chat");
  close.style.position = "absolute";
  close.style.top = "10px";
  close.style.right = "10px";
  close.style.width = "36px";
  close.style.height = "36px";
  close.style.borderRadius = "12px";
  close.style.border = "1px solid rgba(15,23,42,.14)";
  close.style.background = "rgba(255,255,255,.9)";
  close.style.cursor = "pointer";

  panel.appendChild(iframe);
  panel.appendChild(close);
  root.appendChild(panel);
  root.appendChild(btn);
  document.body.appendChild(root);

  function openPanel() { panel.style.display = "block"; }
  function closePanel() { panel.style.display = "none"; }
  function toggle() { panel.style.display === "none" ? openPanel() : closePanel(); }

  btn.addEventListener("click", toggle);
  close.addEventListener("click", closePanel);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closePanel();
  });
})();
"""
    return Response(js, mimetype="application/javascript; charset=utf-8")


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - receives user question, returns AI response.
    
    Expected JSON input: { "question": "user's question" }
    Returns JSON: { "answer": "bot response", "confidence": 0.85 }
    """
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            'answer': 'Please provide a question.',
            'confidence': 0.0
        }), 400
    
    user_question = data['question']
    answer, confidence = get_answer(user_question)

    # Also return the best matched FAQ question for transparency/debugging.
    # This helps beginners understand why a response was chosen.
    ensure_faq_model_uptodate()
    normalized = user_question.strip().lower() if isinstance(user_question, str) else ""
    user_vector = vectorizer.transform([normalized]) if normalized else None
    best_match_question = ""
    if user_vector is not None:
        sims = cosine_similarity(user_vector, question_vectors)[0]
        best_match_question = faq_questions[int(np.argmax(sims))]

    return jsonify({
        'answer': answer,
        'confidence': round(confidence, 2),
        'matched_faq': best_match_question
    })


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Feedback endpoint - stores user feedback (helpful / not helpful).
    
    Expected JSON: { "question": "...", "answer": "...", "feedback": "helpful" or "not_helpful" }
    """
    data = request.get_json()
    
    if not data or 'feedback' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid feedback'}), 400
    
    feedback_type = data['feedback']
    if feedback_type not in ['helpful', 'not_helpful']:
        return jsonify({'status': 'error', 'message': 'Invalid feedback type'}), 400
    
    save_feedback(
        data.get("question", ""),
        data.get("answer", ""),
        feedback_type,
        confidence=data.get("confidence", None),
        matched_faq=data.get("matched_faq", ""),
        comment=data.get("comment", ""),
    )
    
    return jsonify({'status': 'success'})


# -----------------------------------------------------------------------------
# STEP 4: Run the Application
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 50)
    print("AI Customer Support Assistant - College FAQ Bot")
    print("=" * 50)
    print(f"Loaded {len(faq_questions)} FAQs from {FAQ_FILE}")
    print("Starting server at http://127.0.0.1:5000")
    print("=" * 50)
    
    app.run(debug=True, port=5000)

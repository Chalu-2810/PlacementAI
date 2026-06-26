"""
app.py
------
Placement Prediction & Career Guidance System
Flask Backend — fully corrected and production/deployment-ready.

FIXES APPLIED (functional):
  1. [QUIZ BUG] load_json() now extracts the actual question list from the
     nested-dict JSON structure (e.g. {"aptitude": [...]} -> [...]).
     pick_random_questions() is simplified -- it only receives plain lists now.
  2. [DB MISMATCH] save_quiz_scores() now correctly passes total_score to the
     INSERT so the column is never NULL.

FIXES APPLIED (deployment / Render):
  3. [PORT] App now binds to the $PORT environment variable injected by
     Render at runtime instead of a hardcoded port=5000.
  4. [SECRET_KEY] SECRET_KEY is now required from the environment with no
     hardcoded fallback -- app will fail fast at startup if it's missing,
     rather than silently using an insecure default in production.
  5. [DEBUG] debug mode is now driven by the FLASK_DEBUG env var and
     defaults to False, so the Werkzeug debugger/reloader never runs in
     production unless explicitly enabled for local development.
  6. [QUIZ STATE] QUIZ_QUESTIONS is no longer a single process-global dict.
     Under Gunicorn with multiple workers, a global in-memory dict is
     per-worker, so one user's generated quiz could be invisible to (or
     overwritten by) another user handled by a different worker. Quiz
     questions for the current attempt are now stored in the user's own
     Flask session instead, which is safe across workers and users.
"""

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os, json, re, random
import datetime
import database as db
import prediction as pred
from utils.resume_parser import extract_text
from utils.ats_score import calculate_ats, get_ats_breakdown

app = Flask(__name__)

# ------------------------------------------------------------------
# FIX 4: SECRET_KEY must be set via environment variable in production.
# No insecure hardcoded fallback. App fails fast at startup if missing,
# which is preferable to silently running with a guessable default key.
# ------------------------------------------------------------------
app.secret_key = os.environ['SECRET_KEY']

app.jinja_env.globals.update(enumerate=enumerate)


# ============================================================
# ACCESS DECORATORS
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin access required.', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


# ------------------------------------------------------------------
# FIX 1: load_json now UNWRAPS the nested-dict JSON structure.
#
# Every question file looks like one of:
#   {"aptitude": [...]}          <- aptitude files
#   {"data_science": [...]}      <- technical files use domain key
#   {"communication": [...]}     <- communication file
#
# We always want just the plain list inside, regardless of the key name.
# ------------------------------------------------------------------
def load_json(file_path):
    """Load a question JSON file and return a plain list of question dicts."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # If the file is a dict with one key whose value is a list, unwrap it.
        if isinstance(data, dict):
            values = list(data.values())
            if len(values) == 1 and isinstance(values[0], list):
                return values[0]
            # Multiple keys — flatten all lists (shouldn't happen, but safe)
            flat = []
            for v in values:
                if isinstance(v, list):
                    flat.extend(v)
            return flat

        # Already a plain list
        if isinstance(data, list):
            return data

        return []

    except Exception as e:
        print(f"[WARN] Could not load {file_path}: {e}")
        return []


def pick_random_questions(question_list, num=10):
    """Pick `num` random questions from a plain list of question dicts."""
    if not question_list:
        return []
    # Safety: filter to only proper dicts (question objects)
    question_list = [q for q in question_list if isinstance(q, dict)]
    return random.sample(question_list, min(num, len(question_list)))


# Load communication questions once at startup
try:
    communication_questions = load_json("questions/communication.json")
    print(f"[INFO] Loaded {len(communication_questions)} communication questions.")
except Exception:
    communication_questions = []

DOMAIN_FILES = {
    "Data Science"            : {"technical": "questions/technical/ds.json",      "aptitude": "questions/aptitude/ds.json"},
    "Web Development"         : {"technical": "questions/technical/wd.json",      "aptitude": "questions/aptitude/wd.json"},
    "Cybersecurity"           : {"technical": "questions/technical/cs.json",      "aptitude": "questions/aptitude/cs.json"},
    "Cloud Computing"         : {"technical": "questions/technical/cc.json",      "aptitude": "questions/aptitude/cc.json"},
    "DevOps"                  : {"technical": "questions/technical/devops.json",  "aptitude": "questions/aptitude/devops.json"},
    "Internet of Things (IoT)": {"technical": "questions/technical/iot.json",     "aptitude": "questions/aptitude/iot.json"},
    "Blockchain Technology"   : {"technical": "questions/technical/bt.json",      "aptitude": "questions/aptitude/bt.json"},
    "Mobile App Development"  : {"technical": "questions/technical/mad.json",     "aptitude": "questions/aptitude/mad.json"},
}


# ============================================================
# AUTH ROUTES
# ============================================================
@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    domains = db.get_all_domains()
    if request.method == 'POST':
        data    = request.form
        email   = data.get('email', '').strip()

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email address.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        password = data.get('password', '')
        confirm  = data.get('confirm_password', '')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        try:
            db.create_user(
                full_name     = data['full_name'].strip(),
                email         = email,
                password_hash = generate_password_hash(password),
                register_no   = data['register_no'].strip(),
                department    = data['department'].strip(),
                batch_year    = int(data['batch_year'])
            )
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Registration failed: {e}', 'danger')

    return render_template('registration.html', domains=domains, data=None)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = db.get_user_by_email(request.form.get('email', '').strip())
        if user and check_password_hash(user['password'], request.form.get('password', '')):
            session['user_id']  = user['id']
            session['username'] = user['full_name']
            flash(f"Welcome back, {user['full_name']}! 👋", 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ============================================================
# DASHBOARD
# ============================================================
@app.route('/dashboard')
@login_required
def dashboard():
    user_id    = session['user_id']
    user       = db.get_user_by_id(user_id)
    academic   = db.get_academic_details(user_id)
    quiz       = db.get_latest_quiz_scores(user_id)
    prediction = db.get_latest_prediction(user_id)
    return render_template('dashboard.html',
                           user=user, academic=academic,
                           quiz=quiz, prediction=prediction)


# ============================================================
# ACADEMIC DETAILS
# ============================================================
@app.route('/academic', methods=['GET', 'POST'])
@login_required
def academic():
    user_id          = session['user_id']
    domains          = db.get_all_domains()
    academic_details = db.get_academic_details(user_id)

    if request.method == 'POST':
        data = request.form
        try:
            db.save_academic_details(
                user_id   = user_id,
                tenth     = float(data['tenth']),
                twelfth   = float(data['twelfth']),
                cgpa      = float(data['cgpa']),
                backlogs  = int(data['backlogs']),
                domain_id = int(data['domain'])
            )
            flash('Academic details saved!', 'success')
            return redirect(url_for('quiz_setup'))
        except Exception as e:
            flash(f'Error saving details: {e}', 'danger')

    return render_template('academic.html',
                           academic_details=academic_details,
                           domains=domains)


# ============================================================
# QUIZ
# ============================================================
@app.route('/quiz/setup')
@login_required
def quiz_setup():
    academic = db.get_academic_details(session['user_id'])
    domains  = db.get_all_domains()
    return render_template('quiz_setup.html', academic=academic, domains=domains)


@app.route('/generate_quiz', methods=['POST'])
@login_required
def generate_quiz():
    try:
        data = request.get_json()
        print("[generate_quiz] Incoming data:", data)

        domain_id = data.get('domain')
        if not domain_id:
            return jsonify({"error": "Domain missing"}), 400

        domain = db.get_domain_name(int(domain_id))
        print("[generate_quiz] Domain Name:", domain)

        if not domain or domain not in DOMAIN_FILES:
            return jsonify({"error": f"Invalid domain: {domain}"}), 400

        files = DOMAIN_FILES[domain]

        # FIX 1: load_json now returns a plain list — no more nested dict issues
        technical_qs = load_json(files['technical'])
        aptitude_qs  = load_json(files['aptitude'])

        print(f"[generate_quiz] Loaded {len(aptitude_qs)} aptitude, "
              f"{len(technical_qs)} technical, "
              f"{len(communication_questions)} communication questions.")

        quiz = {
            "aptitude"     : pick_random_questions(aptitude_qs, 10),
            "technical"    : pick_random_questions(technical_qs, 10),
            "communication": pick_random_questions(communication_questions, 10),
        }

        # FIX 6: store the generated quiz in the user's own session instead
        # of a process-global dict, so it's correct under multiple Gunicorn
        # workers and isolated per user.
        session['quiz_questions'] = quiz
        return jsonify({"status": "ok"})

    except Exception as e:
        print("🔥 ERROR in generate_quiz:", str(e))
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/quiz', methods=['GET'])
@login_required
def quiz_page():
    quiz = session.get('quiz_questions', {"aptitude": [], "technical": [], "communication": []})
    if not quiz.get('aptitude'):
        flash("Please generate your quiz first by selecting a domain.", "warning")
        return redirect(url_for('quiz_setup'))
    return render_template('quiz.html', quiz=quiz)


@app.route('/submit_quiz', methods=['POST'])
@login_required
def submit_quiz():
    user_id = session['user_id']
    answers = request.form
    quiz_questions = session.get('quiz_questions', {"aptitude": [], "technical": [], "communication": []})

    def score_section(section):
        qs = quiz_questions.get(section, [])
        if not qs:
            return 0

        correct = 0

        for q in qs:
            if not isinstance(q, dict):
                continue

            key = f"{section}_{q.get('id')}"

            user_ans = answers.get(key)

            print("DEBUG:", key, "->", user_ans, "| correct:", q.get('answer'))

            if user_ans == q.get('answer'):
                correct += 1

        return round((correct / len(qs)) * 100, 2)

    apt = score_section('aptitude')
    tech = score_section('technical')
    comm = score_section('communication')

    db.save_quiz_scores(user_id, apt, tech, comm)

    flash(f"Quiz submitted! Aptitude: {apt}% | Technical: {tech}% | Communication: {comm}%", "success")
    return redirect(url_for('resume'))
# ============================================================
@app.route('/resume', methods=['GET', 'POST'])
@login_required
def resume():
    ats_result = session.get('ats_result')
    if request.method == 'POST':
        file = request.files.get('resume_pdf')
        if not file or file.filename == '':
            flash("Please select a PDF file.", "warning")
            return redirect(url_for('resume'))
        if not file.filename.lower().endswith('.pdf'):
            flash("Only PDF files are accepted.", "danger")
            return redirect(url_for('resume'))
        try:
            text      = extract_text(file)
            ats_score = calculate_ats(text)
            breakdown = get_ats_breakdown(text)
            session['resume_score'] = ats_score
            session['ats_result']   = breakdown
            flash(f"Resume analyzed! ATS Score: {ats_score}/100", "success")
            return redirect(url_for('resume'))
        except Exception as e:
            flash(f"Error parsing resume: {e}", "danger")

    return render_template('resume.html',
                           ats_result=ats_result,
                           resume_score=session.get('resume_score', 0))


# ============================================================
# PREDICTION
# ============================================================
@app.route('/predict')
@login_required
def predict():
    user_id  = session['user_id']
    academic = db.get_academic_details(user_id)
    quiz     = db.get_latest_quiz_scores(user_id)

    if not academic:
        flash('Please complete your academic details first.', 'warning')
        return redirect(url_for('academic'))

    if not quiz:
        flash('Please complete the skill assessment quiz first.', 'warning')
        return redirect(url_for('quiz_setup'))

    domain_name = db.get_domain_name(academic['preferred_domain_id'])
    if not domain_name:
        flash("Invalid domain. Please update academic details.", "danger")
        return redirect(url_for('academic'))

    # Build input_data — all fields guaranteed present due to checks above
    input_data = {
        'tenth'        : float(academic['tenth_percentage']),
        'twelfth'      : float(academic['twelfth_percentage']),
        'cgpa'         : float(academic['cgpa']),
        'backlogs'     : int(academic['backlogs']),
        'aptitude'     : float(quiz.get('aptitude_score') or 0),
        'technical'    : float(quiz.get('technical_score') or 0),
        'communication': float(quiz.get('communication_score') or 0),
        'resume_score' : float(session.get('resume_score', 0)),
        'domain'       : domain_name,
    }

    result = pred.predict_placement(input_data)

    # FIX 2: academic_score uses a weighted formula that matches schema intent.
    # skill_score is the average of the three quiz sections (0–100 scale).
    academic_score = round(
        (input_data['tenth'] * 0.2 +
         input_data['twelfth'] * 0.3 +
         input_data['cgpa'] * 10 * 0.5), 2
    )
    skill_score = round(
        (input_data['aptitude'] + input_data['technical'] + input_data['communication']) / 3, 2
    )

    db.save_prediction_results(
        user_id         = user_id,
        academic_score  = academic_score,
        skill_score     = skill_score,
        resume_score    = input_data['resume_score'],
        prediction      = result['prediction'],
        confidence      = result['confidence'] / 100,
        readiness_score = result['readiness_score'],
        readiness_level = result['readiness_level'],
        model_used      = 'Random Forest',
        predicted_domain= domain_name,
    )

    sg = result['skill_gap']
    db.save_skill_gap(
        user_id          = user_id,
        apt_gap          = sg['apt_gap'],
        tech_gap         = sg['tech_gap'],
        comm_gap         = sg['comm_gap'],
        weak_areas_json  = json.dumps(sg['weak_areas']),
        suggestions_json = json.dumps(sg['suggestions']),
    )

    return render_template(
        'result.html',
        result   = result,
        academic = academic,
        quiz     = quiz,
        user     = db.get_user_by_id(user_id),
    )


# ============================================================
# ADMIN
# ============================================================
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin = db.get_admin_by_username(request.form.get('username', '').strip())
        if admin and check_password_hash(admin['password'], request.form.get('password', '')):
            session['admin_id']   = admin['id']
            session['admin_name'] = admin['username']
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_name', None)
    flash('Admin logged out.', 'info')
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    students = db.get_all_students_summary()
    stats    = db.get_placement_stats()
    return render_template('admin_dashboard.html', students=students, stats=stats)


# ============================================================
# API ENDPOINTS (JSON)
# ============================================================
@app.route('/api/domains')
def api_domains():
    return jsonify(db.get_all_domains())


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    # FIX 3: bind to Render's injected $PORT instead of a hardcoded port.
    # FIX 5: debug defaults to False; only enabled locally via FLASK_DEBUG=1.
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
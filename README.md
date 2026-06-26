# 🎓 PlacementAI — Placement Prediction & Career Guidance System

A complete Final Year Project for B.Tech / BE students.  
ML-powered web application that predicts placement probability, analyses skill gaps, recommends job roles, courses, and provides a 90-day career roadmap.

---

## 📁 Project Structure

```
PlacementAI/
├── app.py                  ← Flask application (main entry point)
├── database.py             ← All MySQL database operations
├── prediction.py           ← ML prediction, skill gap, roadmap logic
├── generate_dataset.py     ← Generates synthetic training dataset
├── model_training.py       ← Trains & saves ML models
├── requirements.txt        ← Python dependencies
├── schema.sql              ← MySQL database schema
├── .env.example            ← Environment variable template
│
├── models/                 ← Saved ML model files (.pkl)
│   ├── trained_model.pkl
│   ├── all_models.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
│
├── questions/              ← Quiz question bank (JSON)
│   ├── communication.json
│   ├── aptitude/           ← Domain-wise aptitude questions
│   └── technical/          ← Domain-wise technical questions
│
├── data/
│   └── domains.json        ← Domain list
│
├── dataset/                ← Generated CSV dataset (after running generate_dataset.py)
│
├── utils/
│   ├── resume_parser.py    ← PDF text extraction
│   └── ats_score.py        ← ATS resume scoring (0–100)
│   └──_init_.py
    └──resume.py
    
├── templates/              ← Jinja2 HTML templates
│   ├── base.html
│   ├── login.html
│   ├── registration.html
│   ├── dashboard.html
│   ├── academic.html
│   ├── quiz_setup.html
│   ├── quiz.html
│   ├── resume.html
│   ├── result.html
│   ├── admin_login.html
│   └── admin_dashboard.html
│
└── static/
    └── css/
        └── style.css
```

---

## ⚙️ Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyMuPDF (`fitz`) requires Python ≥ 3.8. If installation fails, the system will fall back to pdfminer.

---

### 2. Set Up MySQL Database

Make sure MySQL is running, then:

```bash
mysql -u root -p < schema.sql
```

This creates the `placement_db` database with all tables and a default admin account.

---

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your MySQL password and secret key
```

---

### 4. Train the ML Model (First-time setup)

```bash
python generate_dataset.py   # Creates dataset/placement_data.csv
python model_training.py     # Trains models and saves to models/
```

> **Skip this step** if the `models/` folder already contains `.pkl` files (pre-trained models are included).

---

### 5. Run the Application

```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🔐 Login Credentials

### Admin
| Username | Password |
|----------|----------|
| `admin`  | (set during schema.sql) — update password in DB |

**To set the admin password:**
```python
from werkzeug.security import generate_password_hash
print(generate_password_hash('your_password'))
```
Then `UPDATE admins SET password='<hash>' WHERE username='admin';`

### Student
Register at `/register` using any email and password.

---

## 🧭 Student Journey (Step-by-Step)

```
Register → Login → Academic Details → Select Domain → Take Quiz (30 Qs)
→ Upload Resume (optional) → Run ML Prediction → View Full Report
```

---

## 🤖 Machine Learning Details

| Feature | Description |
|---------|-------------|
| 10th % | Secondary school marks |
| 12th % | Higher secondary marks |
| CGPA | College CGPA (out of 10) |
| Backlogs | Number of active backlogs |
| Aptitude Score | Quiz score (0–100%) |
| Technical Score | Domain-specific quiz (0–100%) |
| Communication Score | Verbal/written quiz (0–100%) |
| Resume ATS Score | ATS compatibility score (0–100) |
| Domain | Preferred job domain (label encoded) |
| Academic Score | Weighted composite (10th + 12th + CGPA) |
| Skill Score | Weighted composite (Apt + Tech + Comm + ATS) |
| Backlog Penalty | Binary flag (0 backlogs = 1) |

**Models Trained:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM  
**Best Model:** Saved automatically based on highest F1-Score  
**Output:** Placed / Not Placed + Confidence % + Readiness Score

---

## 📊 Features at a Glance

- ✅ Student Registration & Login
- ✅ Academic Details Form (10th, 12th, CGPA, Backlogs, Domain)
- ✅ Domain-wise Quiz (30 questions: Aptitude + Technical + Communication)
- ✅ 30-minute countdown quiz timer
- ✅ Resume PDF Upload + ATS Scoring (0–100)
- ✅ ML Prediction (Placed / Not Placed) with Confidence %
- ✅ Placement Readiness Score (out of 100)
- ✅ Multi-model comparison table
- ✅ Skill Gap Analysis with radar chart
- ✅ Job Role Recommendations per domain
- ✅ Recommended Courses (Free + Paid)
- ✅ Personalised 30-60-90 Day Career Roadmap
- ✅ Admin Dashboard with charts and student summary table
- ✅ Printable result report

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.x |
| Database | MySQL 8.x |
| ML | scikit-learn, joblib, NumPy, pandas |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Bootstrap 5.3, Font Awesome 6, Chart.js 4 |
| Auth | Werkzeug password hashing |

---

## 📝 Sample Test Data

| Field | Value |
|-------|-------|
| 10th % | 85 |
| 12th % | 78 |
| CGPA | 7.5 |
| Backlogs | 0 |
| Domain | Data Science |
| (Quiz scores auto-calculated after quiz) |

---

## 🐛 Troubleshooting

**"Access denied for user root"** → Update DB_PASSWORD in your `.env` file  
**"Model file not found"** → Run `python generate_dataset.py` then `python model_training.py`  
**"No quiz available"** → Go to `/quiz/setup` first and select a domain  
**Resume upload fails** → Ensure `PyMuPDF` is installed: `pip install PyMuPDF`

---

*Built as a Final Year Project | Anna University | 2025*
## admin username:admin
## admin password:Admin@123

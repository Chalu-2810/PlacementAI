"""
prediction.py
-------------
Core prediction engine.
Handles ML inference, readiness scoring, skill gap, domain recommendation, roadmap.
"""

import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# LOAD ARTIFACTS
# ============================================================
MODEL_DIR = 'models'

def _load(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None

model         = _load('trained_model.pkl')
scaler        = _load('scaler.pkl')
label_encoder = _load('label_encoder.pkl')
feature_names = _load('feature_names.pkl')
all_models    = _load('all_models.pkl')

# Average benchmark scores for placed students
PLACED_AVG = {
    'aptitude_score'     : 70.5,
    'technical_score'    : 72.3,
    'communication_score': 68.8
}

# ============================================================
# PREPROCESS INPUT
# ============================================================
def _build_feature_vector(data: dict):
    """
    Builds a feature vector compatible with the trained scaler/model.

    Supports both artifact generations:
      - 11-feature (old): no resume_score column
      - 12-feature (new): resume_score at index 7

    Feature order expected by model_training.py (12-feature):
        0  tenth_percentage
        1  twelfth_percentage
        2  cgpa
        3  backlogs
        4  aptitude_score
        5  technical_score
        6  communication_score
        7  resume_score
        8  domain_encoded
        9  academic_score
        10 skill_score
        11 backlog_penalty

    FIX — skill_score weights must match model_training.py exactly:
        model_training: apt*0.30 + tech*0.40 + comm*0.15 + resume*0.15
        The old prediction.py used apt*0.35 + tech*0.45 + comm*0.20 with no
        resume term — different weights that caused a systematic feature-value
        mismatch at inference time.
    """
    resume_score = float(data.get('resume_score', 0))

    # Domain encoding — handle unseen labels gracefully
    domain = data.get('domain', 'Data Science')
    try:
        domain_encoded = int(label_encoder.transform([domain])[0])
    except Exception:
        domain_encoded = 0  # fallback to first class

    academic_score = (
        float(data['tenth'])   * 0.2 +
        float(data['twelfth']) * 0.2 +
        float(data['cgpa'])    * 10  * 0.6
    )

    backlog_penalty = 1 if int(data['backlogs']) == 0 else 0

    # FIX: use the training-time formula for skill_score.
    # 12-feature model includes resume_score in the weighted composite;
    # 11-feature legacy model omits it (resume not a trained feature there).
    if feature_names and len(feature_names) == 12:
        skill_score = (
            float(data['aptitude'])      * 0.30 +
            float(data['technical'])     * 0.40 +
            float(data['communication']) * 0.15 +
            resume_score                 * 0.15
        )
    else:
        skill_score = (
            float(data['aptitude'])      * 0.35 +
            float(data['technical'])     * 0.45 +
            float(data['communication']) * 0.20
        )

    # Build the base 11-feature vector (legacy column order)
    base_vector = [
        float(data['tenth']),
        float(data['twelfth']),
        float(data['cgpa']),
        float(data['backlogs']),
        float(data['aptitude']),
        float(data['technical']),
        float(data['communication']),
        domain_encoded,
        academic_score,
        skill_score,
        backlog_penalty
    ]

    # For 12-feature models insert resume_score at position 7 so the order
    # matches training: [..., communication, resume_score, domain_encoded, ...]
    if feature_names and len(feature_names) == 12:
        base_vector.insert(7, resume_score)

    # Return as a named DataFrame so sklearn's StandardScaler does not emit
    # "X does not have valid feature names" UserWarnings on every call.
    cols = feature_names if feature_names else list(range(len(base_vector)))
    return pd.DataFrame([base_vector], columns=cols)


# ============================================================
# MAIN PREDICTION
# ============================================================
def predict_placement(data: dict) -> dict:
    X        = _build_feature_vector(data)
    X_scaled = scaler.transform(X)

    prediction_raw   = model.predict(X_scaled)[0]
    prediction_label = "Placed" if prediction_raw == 1 else "Not Placed"

    proba      = model.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)) * 100, 2)

    # All model comparisons
    all_preds = {}
    if all_models:
        for mname, m in all_models.items():
            try:
                p    = m.predict(X_scaled)[0]
                prob = m.predict_proba(X_scaled)[0]
                all_preds[mname] = {
                    'prediction': "Placed" if p == 1 else "Not Placed",
                    'confidence': round(float(max(prob)) * 100, 2)
                }
            except Exception:
                pass

    readiness  = compute_readiness_score(data, confidence)
    skill_gap  = compute_skill_gap(data)
    domain_rec = get_domain_recommendation(data['domain'])
    job_roles  = get_job_roles(data['domain'], prediction_label)
    roadmap    = get_career_roadmap(data['domain'], skill_gap['weak_areas'])
    courses    = get_courses(data['domain'])

    return {
        'prediction'           : prediction_label,
        'confidence'           : confidence,
        'readiness_score'      : readiness['score'],
        'readiness_level'      : readiness['level'],
        'all_model_preds'      : all_preds,
        'skill_gap'            : skill_gap,
        'domain_recommendation': domain_rec,
        'job_roles'            : job_roles,
        'roadmap'              : roadmap,
        'courses'              : courses,
    }


# ============================================================
# READINESS SCORE
# ============================================================
def compute_readiness_score(data: dict, confidence: float) -> dict:
    resume_score = float(data.get('resume_score', 0))

    academic = (
        (float(data['tenth'])   / 100) * 15 +
        (float(data['twelfth']) / 100) * 15 +
        (float(data['cgpa'])    / 10)  * 20
    )  # max 50

    skills = (
        (float(data['aptitude'])      / 100) * 10 +
        (float(data['technical'])     / 100) * 12 +
        (float(data['communication']) / 100) * 8  +
        (resume_score                 / 100) * 5
    )  # max 35

    model_conf      = (confidence / 100) * 15   # max 15
    backlog_penalty = int(data['backlogs']) * 3

    score = round(academic + skills + model_conf - backlog_penalty, 2)
    score = max(0, min(100, score))

    if score >= 75:
        level = "Placement Ready 🎯"
    elif score >= 50:
        level = "Almost There 🚀"
    else:
        level = "Needs Improvement 🔧"

    return {'score': score, 'level': level}


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================
def compute_skill_gap(data: dict) -> dict:
    apt  = float(data['aptitude'])
    tech = float(data['technical'])
    comm = float(data['communication'])
    resume_score = float(data.get('resume_score', 0))

    apt_gap  = round(PLACED_AVG['aptitude_score']      - apt,  2)
    tech_gap = round(PLACED_AVG['technical_score']     - tech, 2)
    comm_gap = round(PLACED_AVG['communication_score'] - comm, 2)

    apt_pct  = round((apt_gap  / PLACED_AVG['aptitude_score'])      * 100, 1) if apt_gap  > 0 else 0
    tech_pct = round((tech_gap / PLACED_AVG['technical_score'])     * 100, 1) if tech_gap > 0 else 0
    comm_pct = round((comm_gap / PLACED_AVG['communication_score']) * 100, 1) if comm_gap > 0 else 0

    weak_areas  = []
    suggestions = []

    if apt_gap > 5:
        weak_areas.append("Aptitude")
        suggestions.append("Practice 30 aptitude questions daily on IndiaBix / PrepInsta.")
    if tech_gap > 5:
        weak_areas.append("Technical Skills")
        suggestions.append("Solve 2 LeetCode / HackerRank problems per day.")
    if comm_gap > 5:
        weak_areas.append("Communication")
        suggestions.append("Join Toastmasters or practice GD/PI mock sessions daily.")
    if resume_score < 60:
        weak_areas.append("Resume Quality")
        suggestions.append("Improve resume with ATS keywords, proper formatting, and quantified achievements.")
    elif resume_score < 75:
        suggestions.append("Enhance resume by adding measurable achievements and strong action verbs.")
    else:
        suggestions.append("Resume is strong. Keep it updated with latest projects and skills.")

    if not weak_areas:
        suggestions.insert(0, "Great! Maintain your current performance level.")

    return {
        'apt_gap'     : apt_gap,
        'tech_gap'    : tech_gap,
        'comm_gap'    : comm_gap,
        'apt_pct'     : apt_pct,
        'tech_pct'    : tech_pct,
        'comm_pct'    : comm_pct,
        'resume_score': resume_score,
        'apt_score'   : apt,
        'tech_score'  : tech,
        'comm_score'  : comm,
        'weak_areas'  : weak_areas,
        'suggestions' : suggestions,
    }


# ============================================================
# DOMAIN DATA
# ============================================================
DOMAIN_DATA = {
    "Data Science": {
        'skills'        : ['Python', 'Pandas/NumPy', 'Machine Learning', 'SQL', 'Data Visualization'],
        'certifications': ['IBM Data Science (Coursera)', 'Google Data Analytics', 'Kaggle Certifications'],
        'platforms'     : ['Kaggle', 'DataCamp', 'Google Colab', 'Analytics Vidhya'],
        'projects'      : ['EDA on Real Dataset', 'ML Prediction Model', 'NLP Sentiment Analysis', 'Sales Dashboard'],
        'resources'     : ['Towards Data Science', 'StatQuest YouTube', 'Hands-On ML Book']
    },
    "Web Development": {
        'skills'        : ['HTML/CSS/JS', 'React.js', 'Node.js', 'MongoDB', 'REST APIs'],
        'certifications': ['Meta Frontend Developer', 'freeCodeCamp Full Stack', 'The Odin Project'],
        'platforms'     : ['Frontend Mentor', 'Scrimba', 'JavaScript30'],
        'projects'      : ['Portfolio Website', 'E-commerce App', 'Blog Platform', 'Chat Application'],
        'resources'     : ['MDN Docs', 'CSS Tricks', 'JavaScript.info']
    },
    "Cybersecurity": {
        'skills'        : ['Network Security', 'Ethical Hacking', 'Cryptography', 'OWASP', 'Linux'],
        'certifications': ['CEH', 'CompTIA Security+', 'Cisco CyberOps'],
        'platforms'     : ['TryHackMe', 'Hack The Box', 'OverTheWire'],
        'projects'      : ['Vulnerability Scanner', 'Password Strength Checker', 'Secure Web App', 'Network Monitor'],
        'resources'     : ['OWASP Guide', 'Cybrary', 'Krebs on Security']
    },
    "Cloud Computing": {
        'skills'        : ['AWS', 'Azure', 'GCP', 'Virtualization', 'Cloud Security'],
        'certifications': ['AWS Solutions Architect', 'Azure Fundamentals', 'Google Cloud Cert'],
        'platforms'     : ['AWS Skill Builder', 'Qwiklabs', 'Microsoft Learn'],
        'projects'      : ['Deploy Web App on AWS', 'Cloud Storage System', 'Serverless Function App', 'CI/CD Pipeline'],
        'resources'     : ['AWS Docs', 'Azure Docs', 'Cloud Academy']
    },
    "DevOps": {
        'skills'        : ['Docker', 'Kubernetes', 'CI/CD', 'Jenkins', 'Linux'],
        'certifications': ['Docker Certified Associate', 'Kubernetes CKAD', 'DevOps on Coursera'],
        'platforms'     : ['Katacoda', 'Play with Docker', 'Kubernetes.io'],
        'projects'      : ['CI/CD Pipeline Setup', 'Dockerized Full-Stack App', 'K8s Deployment', 'Monitoring Dashboard'],
        'resources'     : ['DevOps Roadmap', 'Docker Docs', 'K8s Docs']
    },
    "Internet of Things (IoT)": {
        'skills'        : ['Embedded Systems', 'Arduino', 'Raspberry Pi', 'Sensors', 'MQTT'],
        'certifications': ['Cisco IoT Fundamentals', 'Coursera IoT Specialization'],
        'platforms'     : ['Arduino IDE', 'ThingSpeak', 'Tinkercad'],
        'projects'      : ['Smart Home System', 'IoT Weather Station', 'Smart Irrigation', 'Health Monitor'],
        'resources'     : ['Arduino Docs', 'IoT For Beginners GitHub', 'Raspberry Pi Docs']
    },
    "Blockchain Technology": {
        'skills'        : ['Blockchain Basics', 'Ethereum', 'Smart Contracts', 'Solidity', 'Cryptography'],
        'certifications': ['Blockchain Specialization (Coursera)', 'Ethereum Developer Cert'],
        'platforms'     : ['CryptoZombies', 'Remix IDE', 'Alchemy University'],
        'projects'      : ['Crypto Wallet App', 'Voting DApp', 'NFT Marketplace', 'Supply Chain DApp'],
        'resources'     : ['Ethereum Docs', 'Bitcoin Whitepaper', 'Blockchain Council']
    },
    "Mobile App Development": {
        'skills'        : ['Flutter', 'Android (Kotlin)', 'React Native', 'Firebase', 'UI/UX Design'],
        'certifications': ['Flutter Certification', 'Android Developer Cert', 'React Native Course'],
        'platforms'     : ['Android Studio', 'Flutter Docs', 'Expo'],
        'projects'      : ['To-Do App', 'Real-time Chat App', 'E-commerce App', 'Fitness Tracker App'],
        'resources'     : ['Flutter Docs', 'Android Developers', 'React Native Docs']
    }
}

JOB_ROLES = {
    "Data Science"            : ['Data Analyst', 'Data Scientist', 'ML Engineer', 'AI Engineer', 'BI Analyst'],
    "Web Development"         : ['Frontend Developer', 'Backend Developer', 'Full Stack Developer', 'React Developer'],
    "Cybersecurity"           : ['Security Analyst', 'Ethical Hacker', 'Penetration Tester', 'Security Engineer'],
    "Cloud Computing"         : ['Cloud Engineer', 'Cloud Architect', 'AWS Engineer', 'DevOps Engineer'],
    "DevOps"                  : ['DevOps Engineer', 'Site Reliability Engineer', 'CI/CD Engineer', 'Platform Engineer'],
    "Internet of Things (IoT)": ['IoT Engineer', 'Embedded Systems Engineer', 'Hardware Developer', 'Firmware Engineer'],
    "Blockchain Technology"   : ['Blockchain Developer', 'Smart Contract Engineer', 'Crypto Analyst', 'Web3 Developer'],
    "Mobile App Development"  : ['Android Developer', 'iOS Developer', 'Flutter Developer', 'React Native Developer']
}

COURSES = {
    "Data Science": [
        {"title": "Machine Learning Specialization", "provider": "Coursera / Andrew Ng",  "type": "Paid",       "link": "https://coursera.org/specializations/machine-learning-introduction"},
        {"title": "Data Science Bootcamp",           "provider": "Udemy",                 "type": "Paid",       "link": "https://udemy.com"},
        {"title": "Python for Data Science",         "provider": "freeCodeCamp",          "type": "Free",       "link": "https://freecodecamp.org"},
        {"title": "Kaggle Micro-Courses",            "provider": "Kaggle",                "type": "Free",       "link": "https://kaggle.com/learn"},
        {"title": "IBM Data Science Professional",   "provider": "edX",                   "type": "Free Audit", "link": "https://edx.org"},
    ],
    "Web Development": [
        {"title": "Full Stack Web Dev Bootcamp",  "provider": "Udemy / Angela Yu", "type": "Paid",       "link": "https://udemy.com"},
        {"title": "Meta Frontend Developer",      "provider": "Coursera",          "type": "Paid",       "link": "https://coursera.org"},
        {"title": "Responsive Web Design",        "provider": "freeCodeCamp",      "type": "Free",       "link": "https://freecodecamp.org"},
        {"title": "The Odin Project",             "provider": "Self-paced",        "type": "Free",       "link": "https://theodinproject.com"},
        {"title": "Frontend Basics",              "provider": "Scrimba",           "type": "Free",       "link": "https://scrimba.com"},
    ],
    "Cybersecurity": [
        {"title": "Complete Cyber Security Course",     "provider": "Udemy",         "type": "Paid",      "link": "https://udemy.com"},
        {"title": "Google Cybersecurity Certificate",   "provider": "Coursera",      "type": "Paid",      "link": "https://coursera.org"},
        {"title": "Cybersecurity Essentials",           "provider": "Cisco NetAcad", "type": "Free",      "link": "https://netacad.com"},
        {"title": "TryHackMe Learning Paths",           "provider": "TryHackMe",     "type": "Free Tier", "link": "https://tryhackme.com"},
        {"title": "OWASP Web Security Guide",           "provider": "OWASP",         "type": "Free",      "link": "https://owasp.org"},
    ],
    "Cloud Computing": [
        {"title": "AWS Certified Solutions Architect",  "provider": "Udemy",          "type": "Paid",      "link": "https://udemy.com"},
        {"title": "Google Cloud Professional Cert",     "provider": "Coursera",        "type": "Paid",      "link": "https://coursera.org"},
        {"title": "AWS Cloud Practitioner Essentials",  "provider": "AWS Training",    "type": "Free",      "link": "https://aws.amazon.com/training"},
        {"title": "Azure Fundamentals",                 "provider": "Microsoft Learn", "type": "Free",      "link": "https://learn.microsoft.com"},
        {"title": "GCP Basics on Qwiklabs",             "provider": "Google",          "type": "Free Tier", "link": "https://qwiklabs.com"},
    ],
    "DevOps": [
        {"title": "DevOps Bootcamp",          "provider": "Udemy / TechWorld", "type": "Paid", "link": "https://udemy.com"},
        {"title": "DevOps Engineering on AWS","provider": "Coursera",          "type": "Paid", "link": "https://coursera.org"},
        {"title": "Docker for Beginners",     "provider": "Docker Docs",       "type": "Free", "link": "https://docs.docker.com"},
        {"title": "Kubernetes Basics",        "provider": "Kubernetes.io",     "type": "Free", "link": "https://kubernetes.io/docs/tutorials"},
        {"title": "CI/CD with Jenkins",       "provider": "Jenkins Docs",      "type": "Free", "link": "https://jenkins.io"},
    ],
    "Internet of Things (IoT)": [
        {"title": "IoT Specialization",       "provider": "Coursera / UC San Diego", "type": "Paid",       "link": "https://coursera.org"},
        {"title": "Arduino & IoT Bootcamp",   "provider": "Udemy",                   "type": "Paid",       "link": "https://udemy.com"},
        {"title": "IoT Fundamentals",         "provider": "Cisco NetAcad",           "type": "Free",       "link": "https://netacad.com"},
        {"title": "Raspberry Pi Projects",    "provider": "RPi Foundation",          "type": "Free",       "link": "https://raspberrypi.org"},
        {"title": "IoT Basics",               "provider": "edX",                     "type": "Free Audit", "link": "https://edx.org"},
    ],
    "Blockchain Technology": [
        {"title": "Blockchain Specialization",            "provider": "Coursera / U Buffalo", "type": "Paid",       "link": "https://coursera.org"},
        {"title": "Ethereum & Solidity Complete Course",  "provider": "Udemy",                "type": "Paid",       "link": "https://udemy.com"},
        {"title": "Blockchain Basics",                    "provider": "IBM / edX",            "type": "Free Audit", "link": "https://edx.org"},
        {"title": "CryptoZombies Solidity Course",        "provider": "CryptoZombies",        "type": "Free",       "link": "https://cryptozombies.io"},
        {"title": "Alchemy University Web3",              "provider": "Alchemy",              "type": "Free",       "link": "https://university.alchemy.com"},
    ],
    "Mobile App Development": [
        {"title": "Flutter & Dart Complete Course",       "provider": "Udemy / Angela Yu", "type": "Paid", "link": "https://udemy.com"},
        {"title": "Android Development Specialization",   "provider": "Coursera",          "type": "Paid", "link": "https://coursera.org"},
        {"title": "Android Basics with Compose",          "provider": "Google",            "type": "Free", "link": "https://developer.android.com/courses"},
        {"title": "React Native Tutorial",                "provider": "React Native Docs", "type": "Free", "link": "https://reactnative.dev"},
        {"title": "Flutter Official Docs & Codelabs",     "provider": "Flutter",           "type": "Free", "link": "https://flutter.dev"},
    ],
}


def get_domain_recommendation(domain: str) -> dict:
    return DOMAIN_DATA.get(domain, DOMAIN_DATA['Data Science'])


def get_job_roles(domain: str, prediction: str) -> list:
    roles = JOB_ROLES.get(domain, [])
    return roles[:2] if prediction == "Not Placed" else roles


def get_courses(domain: str) -> list:
    return COURSES.get(domain, COURSES['Data Science'])


def get_career_roadmap(domain: str, weak_areas: list) -> dict:
    base         = DOMAIN_DATA.get(domain, DOMAIN_DATA['Data Science'])
    has_apt_gap  = 'Aptitude'        in weak_areas
    has_tech_gap = 'Technical Skills' in weak_areas
    has_comm_gap = 'Communication'   in weak_areas

    return {
        '30_day_plan': [
            f"Master fundamentals: {', '.join(base['skills'][:2])}",
            "Solve 20 aptitude problems on IndiaBix daily" if has_apt_gap else "Revise core concepts daily",
            f"Complete beginner track on {base['platforms'][0]}",
            "Practice 2 mock technical interviews this week",
        ],
        '60_day_plan': [
            f"Build first project: {base['projects'][0]}",
            f"Earn certification: {base['certifications'][0]}",
            "Practice communication via GD/PI mock sessions" if has_comm_gap else "Contribute to open source",
            f"Learn: {', '.join(base['skills'][2:4])}",
            "Solve medium-level problems on LeetCode/HackerRank" if has_tech_gap else f"Explore {base['platforms'][1]}",
        ],
        '90_day_plan': [
            f"Build & deploy: {base['projects'][1]}",
            "Apply to 5 companies per week via LinkedIn/Naukri",
            f"Complete: {base['certifications'][1]}",
            "Simulate full HR + Technical interview cycle",
            f"Deep dive into: {', '.join(base['skills'][3:])}",
            f"Study key resources: {', '.join(base['resources'][:2])}",
        ]
    }

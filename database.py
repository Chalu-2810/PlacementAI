"""
database.py
-----------
MySQL-only database layer.

FIX APPLIED:
  get_all_students_summary() now uses subqueries to pull only the LATEST
  quiz_score and prediction_result per user, eliminating duplicate rows in
  the admin dashboard when a student has taken multiple quiz attempts.
"""

import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

DB_CONFIG = {
    'host'    : os.getenv('DB_HOST', 'localhost'),
    'user'    : os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'placement_db'),
    'port'    : int(os.getenv('DB_PORT', 3306))
}

print("[DB] Backend: MYSQL")


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def execute_query(query, params=None, fetch=False, fetchone=False):
    conn   = get_connection()
    cursor = conn.cursor(dictionary=True)
    params = params or ()

    try:
        cursor.execute(query, params)

        if fetch:
            return cursor.fetchall()
        elif fetchone:
            return cursor.fetchone()
        else:
            conn.commit()
            return cursor.lastrowid

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        cursor.close()
        conn.close()


# =========================
# USER OPERATIONS
# =========================
def create_user(full_name, email, password_hash, register_no, department, batch_year):
    q = """INSERT INTO users (full_name, email, password, register_no, department, batch_year)
           VALUES (%s, %s, %s, %s, %s, %s)"""
    return execute_query(q, (full_name, email, password_hash, register_no, department, batch_year))


def get_user_by_email(email):
    return execute_query("SELECT * FROM users WHERE email = %s", (email,), fetchone=True)


def get_user_by_id(user_id):
    return execute_query("SELECT * FROM users WHERE id = %s", (user_id,), fetchone=True)


# =========================
# ACADEMIC DETAILS
# =========================
def save_academic_details(user_id, tenth, twelfth, cgpa, backlogs=0, domain_id=1):
    q = """INSERT INTO academic_details
           (user_id, tenth_percentage, twelfth_percentage, cgpa, backlogs, preferred_domain_id)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON DUPLICATE KEY UPDATE
           tenth_percentage=%s, twelfth_percentage=%s, cgpa=%s,
           backlogs=%s, preferred_domain_id=%s"""
    return execute_query(q, (user_id, tenth, twelfth, cgpa, backlogs, domain_id,
                             tenth, twelfth, cgpa, backlogs, domain_id))


def get_academic_details(user_id):
    q = """SELECT a.*, d.domain_name AS domain
           FROM academic_details a
           LEFT JOIN domains d ON a.preferred_domain_id = d.id
           WHERE a.user_id = %s
           ORDER BY a.updated_at DESC LIMIT 1"""
    return execute_query(q, (user_id,), fetchone=True)


def get_all_domains():
    return execute_query("SELECT * FROM domains ORDER BY id", fetch=True)


def get_domain_name(domain_id):
    q      = "SELECT domain_name FROM domains WHERE id = %s"
    result = execute_query(q, (domain_id,), fetchone=True)
    return result['domain_name'] if result else None


# =========================
# QUIZ OPERATIONS
# =========================
def save_quiz_scores(user_id, aptitude, technical, communication):
    total = round((aptitude + technical + communication) / 3, 2)
    q = """
    INSERT INTO quiz_scores
        (user_id, aptitude_score, technical_score, communication_score, total_score)
    VALUES (%s, %s, %s, %s, %s)
    """
    return execute_query(q, (user_id, aptitude, technical, communication, total))


def get_latest_quiz_scores(user_id):
    q = """SELECT * FROM quiz_scores WHERE user_id = %s
           ORDER BY attempt_date DESC LIMIT 1"""
    return execute_query(q, (user_id,), fetchone=True)


# =========================
# PREDICTION OPERATIONS
# =========================
def save_prediction_results(user_id, academic_score, skill_score, resume_score,
                             prediction, confidence, readiness_score, readiness_level,
                             model_used, predicted_domain=None):
    q = """INSERT INTO prediction_results
           (user_id, academic_score, skill_score, resume_score,
            prediction, confidence_score, readiness_score, readiness_level,
            model_used, predicted_domain)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    return execute_query(q, (user_id, academic_score, skill_score, resume_score,
                             prediction, confidence, readiness_score, readiness_level,
                             model_used, predicted_domain))


def get_latest_prediction(user_id):
    q = """SELECT * FROM prediction_results WHERE user_id = %s
           ORDER BY predicted_at DESC LIMIT 1"""
    return execute_query(q, (user_id,), fetchone=True)


# =========================
# SKILL GAP
# =========================
def save_skill_gap(user_id, apt_gap, tech_gap, comm_gap, weak_areas_json, suggestions_json):
    q = """INSERT INTO skill_gap_analysis
           (user_id, apt_gap, tech_gap, comm_gap, weak_areas, suggestions)
           VALUES (%s, %s, %s, %s, %s, %s)"""
    return execute_query(q, (user_id, apt_gap, tech_gap, comm_gap, weak_areas_json, suggestions_json))


# =========================
# ADMIN
# =========================
def get_admin_by_username(username):
    return execute_query("SELECT * FROM admins WHERE username = %s", (username,), fetchone=True)


def get_all_students_summary():
    """
    FIX: Subqueries ensure only the LATEST quiz attempt and latest prediction
    are joined per user, preventing duplicate rows in the admin dashboard.
    """
    q = """
    SELECT
        u.id, u.full_name, u.register_no, u.department, u.batch_year,
        a.tenth_percentage, a.twelfth_percentage, a.cgpa, a.backlogs,
        d.domain_name AS domain,
        q.aptitude_score, q.technical_score, q.communication_score, q.total_score,
        p.prediction, p.confidence_score, p.readiness_score, p.readiness_level
    FROM users u
    LEFT JOIN academic_details a ON u.id = a.user_id
    LEFT JOIN domains d ON a.preferred_domain_id = d.id
    LEFT JOIN (
        SELECT qs.*
        FROM quiz_scores qs
        INNER JOIN (
            SELECT user_id, MAX(id) AS max_id
            FROM quiz_scores
            GROUP BY user_id
        ) latest_q ON qs.id = latest_q.max_id
    ) q ON u.id = q.user_id
    LEFT JOIN (
        SELECT pr.*
        FROM prediction_results pr
        INNER JOIN (
            SELECT user_id, MAX(id) AS max_id
            FROM prediction_results
            GROUP BY user_id
        ) latest_p ON pr.id = latest_p.max_id
    ) p ON u.id = p.user_id
    ORDER BY u.created_at DESC
    """
    return execute_query(q, fetch=True)


def get_placement_stats():
    q = """
    SELECT prediction, COUNT(*) AS count,
           AVG(confidence_score) AS avg_confidence,
           AVG(readiness_score)  AS avg_readiness
    FROM prediction_results
    GROUP BY prediction
    """
    return execute_query(q, fetch=True)

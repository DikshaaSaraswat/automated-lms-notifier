import re
import json
import csv
import time
import os
import urllib3
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from plyer import notification
import getpass

# ML imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://lms.bennett.edu.in"
LOGIN_URL = f"{BASE_URL}/login/index.php"

SNAPSHOT_FILE = "lms_latest.json"
ML_CSV = "lms_features.csv"

# ----------------- Helper Functions -----------------

def parse_due_date(text):
    if not text:
        return None
    try:
        cleaned = re.sub(
            r"(?i)(due|submission closes|closes on|closes|deadline|due:?)[:\-]?",
            "", text).strip()
        return dateparser.parse(cleaned, fuzzy=True)
    except Exception:
        return None

def compute_urgency(due_dt, submitted=False):
    if submitted:
        return "Submitted"
    if not due_dt:
        return "Later"
    now = datetime.now()
    diff = (due_dt - now).total_seconds()
    if diff <= 0:
        return "Gone"
    if diff <= 15 * 60:
        return "Due in <15 min"
    if due_dt.date() == now.date():
        return "Urgent"
    days_left = (due_dt.date() - now.date()).days
    if days_left <= 2:
        return "Soon"
    return "Later"

def safe_get(session, url, max_retries=4, backoff=1.0):
    for i in range(max_retries):
        try:
            resp = session.get(url, timeout=40, verify=False)
            return resp
        except Exception:
            time.sleep(backoff)
            backoff *= 1.5
    print(f"⚠️ Network slow or LMS down for: {url}")
    return None

def extract_course_links(soup):
    links = set()
    for a in soup.select("a[href*='course/view.php']"):
        href = a.get("href")
        if href:
            if href.startswith("/"):
                href = BASE_URL + href
            if "course/view.php" in href:
                links.add(href)
    return list(links)

def fetch_all_course_links(session):
    all_links = set()
    try:
        dashboard_resp = safe_get(session, f"{BASE_URL}/my/")
        if dashboard_resp and dashboard_resp.text:
            sesskey_match = re.search(r"sesskey=(\w+)", dashboard_resp.text)
            sesskey = sesskey_match.group(1) if sesskey_match else None
            if sesskey:
                ajax_url = f"{BASE_URL}/lib/ajax/service.php?sesskey={sesskey}&info=core_course_get_enrolled_courses_by_timeline_classification"
                payload = [{"index": 0, "methodname": "core_course_get_enrolled_courses_by_timeline_classification",
                            "args": {"classification": "all", "limit": 0, "offset": 0}}]
                try:
                    r = session.post(ajax_url, json=payload, timeout=30, verify=False)
                    data = r.json()
                    if data and isinstance(data, list) and "data" in data[0]:
                        for course in data[0]["data"]["courses"]:
                            cid = course.get("id")
                            if cid:
                                all_links.add(f"{BASE_URL}/course/view.php?id={cid}")
                except Exception:
                    pass
    except Exception:
        pass
    if not all_links:
        for path in ["/my/", "/my/courses.php"]:
            resp = safe_get(session, f"{BASE_URL}{path}")
            if resp and resp.text:
                soup = BeautifulSoup(resp.text, "html.parser")
                all_links.update(extract_course_links(soup))
    print(f"📘 Found {len(all_links)} total course links.")
    return list(all_links)

def detect_submission_status_from_assign_page(session, assign_url):
    resp = safe_get(session, assign_url)
    if not resp or not resp.text:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    if re.search(r"Submitted\s+for\s+grading", text, re.I):
        return "submitted"
    if re.search(r"\bSubmitted\b", text, re.I) and not re.search(r"No\s+submission", text, re.I):
        return "submitted"
    if re.search(r"No\s+submission", text, re.I) or re.search(r"Not\s+submitted", text, re.I):
        return "not_submitted"
    if re.search(r"\bDraft\b", text, re.I) or re.search(r"Draft\s+submission", text, re.I):
        return "draft"
    try:
        for th in soup.find_all(['th', 'td']):
            if th.string and 'submission status' in th.string.lower():
                sibling = th.find_next_sibling(['td', 'th'])
                if sibling:
                    s = sibling.get_text(" ", strip=True)
                    if 'submitted' in s.lower():
                        return "submitted"
                    if 'no submission' in s.lower() or 'not submitted' in s.lower():
                        return "not_submitted"
    except Exception:
        pass
    return None

def extract_quiz_times_from_quiz_page(session, quiz_url):
    resp = safe_get(session, quiz_url)
    if not resp or not resp.text:
        return (None, None)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    open_dt = None
    close_dt = None
    m_open = re.search(r"(?i)(Open(?:s|ed)?[:\s]*)([A-Za-z0-9,:\s\-\/]+(?:AM|PM|am|pm)?)", text)
    m_close = re.search(r"(?i)(Close(?:s|d)?[:\s]*)([A-Za-z0-9,:\s\-\/]+(?:AM|PM|am|pm)?)", text)
    if not m_open:
        m_open = re.search(r"(?i)(Starts?[:\s]*)([A-Za-z0-9,:\s\-\/]+(?:AM|PM|am|pm)?)", text)
    if not m_close:
        m_close = re.search(r"(?i)(Ends?[:\s]*)([A-Za-z0-9,:\s\-\/]+(?:AM|PM|am|pm)?)", text)
    try:
        if m_open:
            open_dt = parse_due_date(m_open.group(2))
        if m_close:
            close_dt = parse_due_date(m_close.group(2))
    except Exception:
        pass
    return (open_dt, close_dt)

def extract_activities(session, soup):
    valid_keywords = ["assign", "quiz", "submission", "announce", "assessment", "test", "exam"]
    activities = []
    for li in soup.find_all("li", class_=lambda c: c and "activity" in c):
        raw_text = li.get_text(" ", strip=True)
        low = raw_text.lower()
        if not any(k in low for k in valid_keywords):
            continue
        a = li.find("a", href=True)
        href = a["href"] if a else None
        if href and href.startswith("/"):
            href = BASE_URL + href
        title_el = li.find("span", class_="instancename") or a or li.find(['h3','h2'])
        title = title_el.get_text(strip=True) if title_el else raw_text[:80]
        if any(x in title.lower() for x in ["forum", "file", "resource", "solution", "slot"]):
            continue
        due_match = re.search(r"(?i)(Due[:\s]*[^\n\r|]+(?:, \s*\d{4})?)", raw_text)
        due_text = due_match.group(0) if due_match else None
        due_dt = parse_due_date(due_text) if due_text else None
        typ = "activity"
        if href and "/mod/assign/" in href:
            typ = "assignment"
        elif href and "/mod/quiz/" in href:
            typ = "quiz"
        elif "announce" in low or "announcement" in low:
            typ = "announcement"
        elif "submission" in low:
            typ = "submission"
        item = {
            "title": title,
            "type": typ,
            "url": href,
            "due": due_text or "No due date found",
            "due_dt": due_dt.isoformat() if due_dt else None,
            "submitted": None,
            "open_dt": None,
            "close_dt": None,
            "urgency": None
        }
        if typ == "assignment" and href:
            try:
                status = detect_submission_status_from_assign_page(session, href)
                if status == "submitted":
                    item["submitted"] = True
                elif status in ("not_submitted", "draft"):
                    item["submitted"] = False
                else:
                    item["submitted"] = None
            except Exception:
                item["submitted"] = None
        if typ == "quiz" and href:
            try:
                open_dt, close_dt = extract_quiz_times_from_quiz_page(session, href)
                if open_dt:
                    item["open_dt"] = open_dt.isoformat()
                if close_dt:
                    item["close_dt"] = close_dt.isoformat()
                if not item["due_dt"] and close_dt:
                    item["due_dt"] = close_dt.isoformat()
            except Exception:
                pass
        submitted_flag = item["submitted"] is True
        dt_for_urgency = None
        if item["due_dt"]:
            try:
                dt_for_urgency = dateparser.parse(item["due_dt"])
            except Exception:
                dt_for_urgency = None
        if typ == "quiz" and item["close_dt"]:
            try:
                dt_for_urgency = dateparser.parse(item["close_dt"])
            except Exception:
                pass
        item["urgency"] = compute_urgency(dt_for_urgency, submitted=submitted_flag)
        activities.append(item)
    seen = set()
    unique = []
    for it in activities:
        key = (it["title"], it["type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return unique

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def notify(title, message, timeout=8):
    icon_path = r"C:\Users\Dikshya\Desktop\SML Project\Project code\logo.ico"

    # Verify icon exists → prevents "Could not load icon" spam
    if not os.path.exists(icon_path):
        print(f"[ERROR] Icon not found at: {icon_path}")
        return

    # Windows hard title limit (UTF-16)
    safe_title = str(title)
    if len(safe_title.encode("utf-16-le")) > 64 * 2:
        while len(safe_title.encode("utf-16-le")) > 64 * 2:
            safe_title = safe_title[:-1]

    # Body limit
    safe_message = str(message)
    if len(safe_message.encode("utf-16-le")) > 255 * 2:
        safe_message = safe_message[:255]

    try:
        notification.notify(
            title=safe_title,
            message=safe_message,
            app_name="LMS",     # <<< HERE (shows "LMS" instead of Python)
            app_icon=icon_path,
            timeout=timeout
        )
    except Exception as e:
        print(f"[notify error] {safe_title}: {safe_message}\n{e}")


def display_summary_all(grouped):
    print("\n================= 📢 LMS Summary (Demo: All Items) =================\n")
    now = datetime.now()
    for cname, acts in grouped.items():
        print(f"📘 Course: {cname}")
        for a in acts:
            name = a.get("title") or a.get("name") or "Unknown Task"
            typ = a.get("type", "")
            urgency = a.get("urgency", "")
            sub = a.get("submitted", None)
            due_text = a.get("due") or "No due date"
            status_str = "Submitted" if sub is True else ("Not Submitted" if sub is False else "Unknown")
            print_str = f"   🧩 {name} | Type: {typ} | Due: {due_text} | Urgency: {urgency} | Status: {status_str}"
            open_info = a.get("open_dt")
            close_info = a.get("close_dt")
            if open_info or close_info:
                print_str += f" | Open: {open_info or '-'} | Close: {close_info or '-'}"
            print(print_str)
        print("")

# ----------------- CSV Export -----------------
def export_ml_csv(snapshot, path=ML_CSV):
    rows = []
    now = datetime.now()
    for course, acts in snapshot.items():
        for a in acts:
            due_dt = None
            try:
                due_dt = dateparser.parse(a["due_dt"]) if a.get("due_dt") else None
            except Exception:
                due_dt = None
            hours_left = None
            days_left = None
            if due_dt:
                delta = (due_dt - now)
                hours_left = round(delta.total_seconds() / 3600, 2)
                days_left = (due_dt.date() - now.date()).days
            submitted_flag = 1 if a.get("submitted") is True else (0 if a.get("submitted") is False else -1)
            rows.append({
                "course": course,
                "title": a.get("title"),
                "type": a.get("type"),
                "submitted": submitted_flag,
                "urgency": a.get("urgency"),
                "hours_left": hours_left,
                "days_left": days_left,
                "has_due": 1 if a.get("due_dt") else 0
            })
    fieldnames = ["course", "title", "type", "submitted", "urgency", "hours_left", "days_left", "has_due"]
    try:
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    except Exception as e:
        print("⚠️ Could not write ML CSV:", e)

# ----------------- ML Module -----------------
def ml_predict_submission(csv_path=ML_CSV):
    try:
        df = pd.read_csv(csv_path)
        le = LabelEncoder()
        df['type_enc'] = le.fit_transform(df['type'])
        features = ['hours_left', 'days_left', 'has_due', 'type_enc']
        df = df.dropna(subset=features + ['submitted'])
        X = df[features]
        y = df['submitted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n📊 ML Prediction Accuracy: {acc*100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, preds))
        df['submission_risk'] = model.predict_proba(X)[:,0]
        print("\nSample Submission Risk:\n", df[['title','course','submission_risk']].head())
        return model, le
    except Exception as e:
        print("⚠️ ML module error:", e)
        return None, None

def run():
    print("Enter your Bennett LMS credentials.")
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    session = requests.Session()
    resp = safe_get(session, LOGIN_URL)
    if not resp or not resp.text:
        print("❌ LMS did not respond. Try again in 1–2 minutes.")
        return
    soup = BeautifulSoup(resp.text, "html.parser")
    token_el = soup.find("input", {"name": "logintoken"})
    token = token_el["value"] if token_el else ""
    payload = {"username": username, "password": password, "logintoken": token}
    try:
        r = session.post(LOGIN_URL, data=payload, timeout=30, verify=False)
    except Exception as e:
        print("⚠️ Login request failed:", e)
        return
    if "login" in r.url:
        print("❌ Login failed! Check credentials.")
        return
    print("✅ Logged in successfully!")

    grouped = {}
    links = fetch_all_course_links(session)
    for link in links:
        page = safe_get(session, link)
        if not page or not page.text:
            print(f"⚠️ Skipped (LMS slow) → {link}")
            continue
        csoup = BeautifulSoup(page.text, "html.parser")
        title_tag = csoup.find("h1") or csoup.find("title")
        course_name = title_tag.get_text(strip=True) if title_tag else link
        activities = extract_activities(session, csoup)
        grouped[course_name] = activities

    save_json_file(SNAPSHOT_FILE, grouped)
    export_ml_csv(grouped, ML_CSV)

    # Run ML predictions and attach submission risk to each task
    ml_model, le_type = ml_predict_submission(ML_CSV)
    ml_data = pd.read_csv(ML_CSV)
    for course, tasks in grouped.items():
        for t in tasks:
            match = ml_data[
                (ml_data['course'] == course) & (ml_data['title'] == t.get('title'))
            ]
            if not match.empty:
                t['submission_risk'] = float(match.iloc[0].get('submission_risk', 0))
            else:
                t['submission_risk'] = None

    print("\n📢 Sending notifications for ALL items with ML insights...")
    for course, tasks in grouped.items():
        for t in tasks:
            urgency = t.get("urgency", "Later")
            typ = t.get("type", "")
            submitted = t.get("submitted", None)
            risk = t.get("submission_risk", None)  # ML prediction risk
            due_text = t.get("due", "No due date")
            due_dt = None
            if t.get("due_dt"):
                try:
                    due_dt = dateparser.parse(t.get("due_dt"))
                except:
                    due_dt = None

            # Compute time left
            time_left_str = ""
            if due_dt:
                delta = due_dt - datetime.now()
                if delta.total_seconds() > 0:
                    days = delta.days
                    hours = delta.seconds // 3600
                    time_left_str = f"{days}d {hours}h left"

            reason = ""
            if typ == "assignment" and submitted is True:
                reason = " (Already Submitted/Demo)"
            elif urgency == "Gone":
                reason = " (Deadline Passed/Demo)"
            elif urgency == "Later" and (submitted is None or submitted is False):
                reason = " (Upcoming/Not Submitted)"

            name = t.get("title") or t.get("name") or "Unknown Task"
            # Build the notification message
            title = f"{urgency} - {name}{reason}"[:64]
            message_parts = [f"{course}", f"Due: {due_text}"]
            if time_left_str:
                message_parts.append(f"Time Left: {time_left_str}")
            if risk is not None:
                message_parts.append(f"Submission Risk: {risk*100:.0f}%")
            message = "\n".join(message_parts)[:200]

            notify(title, message, timeout=8)

    display_summary_all(grouped)
    print(f"\n💾 Course summary saved to {SNAPSHOT_FILE}")
    print(f"💾 ML CSV exported to {ML_CSV}")
    print("📢 Notifications sent for ALL items.")

    
if __name__ == "__main__":
    run()


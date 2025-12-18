# AlertX – Automated LMS Deadline Notifier with ML

AlertX is a Python-based automation and machine learning system that logs into the LMS,
extracts assignments and quizzes, predicts submission risk, and sends real-time desktop
notifications to help students never miss a deadline.

---

## 🚀 Features
- Automated login and navigation of LMS
- Extraction of assignments, quizzes, and deadlines
- Urgency detection based on due date and submission status
- Machine Learning model (Random Forest) to predict submission risk
- Real-time desktop notifications with deadline and risk alerts
- Export of structured data in JSON and CSV formats

---

## 🛠 Tech Stack
- **Language:** Python  
- **Web Scraping:** BeautifulSoup, Requests  
- **Machine Learning:** scikit-learn (Random Forest)  
- **Data Processing:** Pandas  
- **Notifications:** Plyer  

---

## 📂 Project Structure
alertx-lms-notifier/
│
├── alertx.py
├── requirements.txt
├── lms_latest.json
├── lms_features.csv


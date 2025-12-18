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

## ▶️ How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/DikshaaSaraswat/automated-lms-notifier.git


2. **Install dependencies:**

```bash
pip install -r requirements.txt


3. **Run AlertX:**

```bash
python alertx.py


## 🔐 Security Note

User credentials are taken securely at runtime using input prompts and are **not stored** anywhere in the code or files.


## 📊 Machine Learning Details

- **Model:** Random Forest Classifier  
- **Train-test split:** 80/20  
- **Features used:** time left, days left, activity type, due-date availability  
- **Output:** submission risk probability for each task

## 🎯 Use Case

AlertX helps students manage academic workload efficiently by predicting which assignments are most at risk of late submission and notifying them in advance.


## 👩‍💻 Author

**Diksha Saraswat**  
B.Tech CSE Student | AI/ML & Python Enthusiast


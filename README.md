\# Ghost Job Posting Detector



An unsupervised anomaly detection system built on 119K+ LinkedIn job postings

to flag potential "ghost job" listings — postings with high visibility but

suspiciously low application conversion rates.



\## What it does

\- \*\*Overview page\*\*: dashboard showing dataset stats, apply/view ratio

&#x20; distribution, top companies, and flagged ghost-job candidates

\- \*\*Detector page\*\*: enter any job posting's stats and get an instant

&#x20; ghost-job pattern check



\## How it works

\- Algorithm: Isolation Forest (unsupervised anomaly detection)

\- Features: apply/view ratio, log-scaled views, repost count,

&#x20; skill count, salary disclosed flag

\- Validated against DBSCAN and explained using SHAP feature importance



\## Dataset

LinkedIn Job Postings — Kaggle (118,644 rows, Apr 2024)



\## Tech stack

Python · scikit-learn · pandas · Streamlit



\## Live demo

\[Link will appear here after Streamlit Cloud deployment]


# 🧠 Leukemia Diagnostic Assistant

This is a project I'm building to support **clinicians** in the **early detection** and **treatment guidance** for leukemia patients.

## 🔍 What It Does

- Upload a **blood smear image** to check for signs of leukemia
- Enter **basic patient info** (age, gender, diagnosis)
- Get a **risk summary** based on similar patients in the **TCGA-LAML** dataset
- Ask questions through a **chatbot powered by ChatGPT (GPT-3.5-turbo)**

## ⚙️ Under the Hood

- ML model trained on the **C-NMC 2019 dataset** for image classification
- Cleaned clinical data from **TCGA-LAML** for patient matching and outcome estimates
- Chatbot runs on **OpenAI GPT-3.5-turbo** to interpret input and explain results

## 🛠️ Tech Stack

- Python + Streamlit
- PyTorch (for the image model)
- Pandas (for clinical data lookup)
- OpenAI API (ChatGPT integration)

---

This is an early-stage tool aimed at making leukemia detection more accessible and data-driven.

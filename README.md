# 🧠 Leukemia Diagnostic Assistant

This is a project I'm building to support **clinicians** in the **early detection** and **treatment guidance** for leukemia patients.

## 🔍 What It Does

- Upload a **blood smear image** to check for signs of leukemia  
- Enter **basic patient info** (age, gender, diagnosis)  
- Get a **risk summary** based on similar patients in the **TCGA-LAML** dataset  
- Ask questions through a **chatbot powered by LLaMA 3.1** with **function-calling** to interpret and explain model outputs  

## ⚙️ Under the Hood

- ML model trained on the **C-NMC 2019 dataset** for image classification  
- Cleaned and transformed data from **TCGA-LAML** for patient matching and outcome estimation  
- Clinical risk prediction using both **SVM** and **Histogram Gradient Boosting (HGB)** to reduce bias and improve interpretability  
- Chatbot powered by **LLaMA 3.1** and **LangChain**, using structured **function-calling** to trigger predictions and generate explanations  

## 🛠️ Tech Stack

- Python + Flask (backend API)  
- Vue 3 + Vite (frontend UI)  
- PyTorch (for the CNN image model)  
- scikit-learn (for SVM and HGB clinical models)  
- Pandas (for clinical data processing)  
- LangChain + LLaMA 3.1 (chatbot and tool interface)  

---

This is an early-stage tool aimed at making leukemia detection more **accessible**, **explainable**, and **data-driven**.

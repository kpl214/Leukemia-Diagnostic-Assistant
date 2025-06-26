# 🧠 Leukemia Diagnostic Assistant

This is a project I'm building to support **clinicians** in the **early detection** and **treatment guidance** for leukemia patients.


![Project Prototype Image](https://github.com/user-attachments/assets/6264eaa1-524a-4191-a29c-f4b700d61d7e)


## 🔍 What It Does

- Upload a **histopathology image** (microscopic tissue sample) to check for signs of leukemia  
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

## 🐳 Docker Deployment

This project includes all the files you need to build the app from source using Docker — however, if your goal is to get the app running quickly without waiting for builds or downloading packages, we’ve also included a prebuilt Docker image as a .tar file.

Use the prebuilt image: leukemia-app.tar

1. Clone the repository
   `git clone https://github.com/kpl214/leukemia-diagnostic-assistant.git`
   `cd leukemia-diagnostic-assistant`

2. Load the prebuilt Docker image
   `docker load -i leukemia-app.tar`

3. Run the container
   `docker run -d -p 80:80 leukemia-diagnostic-assistant-nginx:latest`

4. Visit the app
   `http://localhost`

This is the **fastest and easiest way** to try the app - no setup or build steps required.

If you wish to build the image yourself for development purposes (may take longer to use app), utilize:
`docker compose up --build` and visit `http://localhost:5173`

---
Note:

I am still working to upload all relevant files when they are deemed fit for public use.

Most data files will not be uploaded due to sensitive information, but can be downloaded online for project use.
  - Clinical Data: https://portal.gdc.cancer.gov/ -> Projects -> Search "TCGA-LAML" -> Download and extract TSV files
  - Histopathology Images for Training: https://www.cancerimagingarchive.net/collection/c-nmc-2019/

This is an early-stage tool aimed at making leukemia detection more **accessible**, **explainable**, and **data-driven**.

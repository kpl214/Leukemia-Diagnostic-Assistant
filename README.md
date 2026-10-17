# 🧠 Leukemia Diagnostic Assistant

This is a project I'm building to support **clinicians** in the **early detection** and **treatment guidance** for leukemia patients.


![Project Prototype Image](https://github.com/user-attachments/assets/6264eaa1-524a-4191-a29c-f4b700d61d7e)


## 🔍 What It Does

- **Upload a histopathology image** (a blood-smear cell image) and get it classified as **normal** or **leukemic**, with a confidence score and a **Grad-CAM heatmap** showing which part of the image actually drove the call.
- **Describe a patient in plain language** — age, diagnosis, and optionally gender/race — and get a calibrated **recurrence/mortality risk estimate**. The assistant explains *why*: it names the specific factors that drove that patient's result, and tells you if it had to assume a missing detail instead of quietly guessing.
- **Switch between two chat models**: a general-purpose model with full tool access (it can actually run the risk calculator and image classifier), and a medical-domain model for plain conversational Q&A that has no tool access on this endpoint and will tell you to switch back if you ask it for a specific number.
- **Multiple conversations**, managed from a sidebar — start a new chat, rename or delete one, switch between them. Everything persists in the browser across a page refresh.
- Assistant responses render **Markdown and syntax-highlighted code**, not plain text.

## ⚙️ Under the Hood

### The calibrated clinical risk model

Trained on clinical/demographic/outcome data from the NIH's **TCGA-LAML** cohort (Genomic Data Commons). A patient is labeled high-risk if a follow-up visit recorded disease progression/recurrence, or they died within two years of diagnosis.

It predicts from four features — **age at diagnosis, gender, race, and primary diagnosis subtype** — deliberately limited to things a clinician actually knows at diagnosis, before any outcome is known.

- **Base classifier**: a `HistGradientBoostingClassifier` — a histogram-based gradient-boosted ensemble of decision trees. Trees are built one at a time, each one trained to correct the errors of every tree built before it, with each tree's contribution scaled down (`learning_rate`) so no single tree dominates.
- **Calibration**: the base classifier is wrapped in a `CalibratedClassifierCV` using isotonic regression over 5-fold cross-validation. Raw gradient-boosting scores aren't reliable probabilities on their own — calibration maps them so that a predicted "70% risk" corresponds to roughly 70% real-world frequency among similar patients, not just a ranking score.
- **Explainability**: every prediction also runs through SHAP (SHapley Additive exPlanations), computing a fresh, per-patient breakdown of which features pushed that specific result toward higher or lower risk — not a single static importance ranking.
- Currently scores an AUROC of ~0.76 on held-out data.

### The CNN image classifier

A **ResNet18** (pretrained on ImageNet) fine-tuned via transfer learning on the **C-NMC 2019** dataset — single blood-smear cell images labeled normal (`hem`) or leukemic (`all`) by the ISBI 2019 challenge organizers. It outputs a binary classification with a confidence score.

Every classification also runs **Grad-CAM**: the predicted class is backpropagated through the network's last convolutional layer to produce a heatmap of which pixels actually influenced the decision, overlaid on the original image.

### The chatbot layer

Built with **LangChain**'s tool-calling agent pattern. The primary model (an OpenAI-compatible Llama endpoint) decides, from the conversation, whether a question needs a numeric prediction at all — and if so, which tool to call — then explains the result in plain, clinically-worded language with a disclaimer. The secondary model (a medical-domain model) runs as a plain conversational chain with no tool access, for general questions.

## 🛠️ Tech Stack

**Backend**
- Python + Flask (API), Gunicorn (production WSGI server)
- PyTorch + torchvision — CNN model, transfer learning, Grad-CAM
- scikit-learn — `HistGradientBoostingClassifier` + `CalibratedClassifierCV`
- SHAP — per-prediction feature explanations
- pandas — clinical data cleaning and merging
- LangChain + `langchain-openai` — the tool-calling agent and chat model integration

**Frontend**
- Vue 3 + Vite
- `marked` + `highlight.js` — Markdown rendering and code syntax highlighting
- DOMPurify — sanitizing model-generated HTML before it's rendered

**Deployment**
- Docker + Docker Compose (separate dev and production configs)
- nginx — production reverse proxy, serves the built frontend
- Gunicorn — production WSGI server for the Flask API

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

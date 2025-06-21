
from typing import Optional
import pandas as pd
import joblib
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator
import torch
from torchvision import transforms
from PIL import Image

#model = joblib.load("../clinical-data/simplified_svm_model.pkl")
bundle = joblib.load("../clinical-data/hgb_calibrated_bundle.pkl")
preprocessor = bundle["preprocessor"]
model = bundle["model"]
print("Loaded model classes_:", model.classes_)

cnn_model = torch.load(
    "../image_training_data/tissue_cnn_model.pth",
    map_location=torch.device("cpu"),
    weights_only=False,
)
cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
class_names = ["normal", "leukemic"]


def _predict_clinical_risk_internal(
    age_at_diagnosis: int,
    primary_diagnosis: str,
    progression_or_recurrence: str,
    vital_status: Optional[str] = None,
    gender: Optional[str] = None,
    race: Optional[str] = None,
) -> str:
    df = pd.DataFrame([{
        "diagnoses.age_at_diagnosis": age_at_diagnosis,
        "diagnoses.primary_diagnosis": primary_diagnosis,
        "diagnoses.progression_or_recurrence": progression_or_recurrence,
        "demographic.vital_status": vital_status,
        "demographic.gender": gender,
        "demographic.race": race,
    }])

    X_pre = preprocessor.transform(df)
    prob = model.predict_proba(X_pre)[0]
    low = float(prob[0])
    high = float(prob[1])
    print("🧪 Raw probabilities:", low, high)

    return {
        "low_risk_probability": round(low * 100, 1),
        "high_risk_probability": round(high * 100, 1),
    }


class ClinicalRiskInput(BaseModel):
    """Schema for the predict_clinical_risk tool."""
    age_at_diagnosis: int = Field(
        ...,
        description = "Patient age at diagnosis in years, e.g. 59"
    )
    primary_diagnosis: str = Field(
        ...,
        description = "Primary diagnosis, e.g. 'Acute Myeloid Leukemia'"
    )
    progression_or_recurrence: str = Field(
        ...,
        description = "'Yes' if the disease has progressed or recurred, otherwise 'No'"
    )
    vital_status: Optional[str] = Field(
        None,
        description = "'Alive' or 'Dead'; optional at prediction time"
    )
    gender: Optional[str] = Field(
        None,
        description = "Patient gender, e.g. 'Male', 'Female', etc."
    )
    race: Optional[str] = Field(
        None,
        description = "Patient race/ethnicity, e.g. 'Black or African American'"
    )

    # normalise common boolean / text variants
    @field_validator("progression_or_recurrence", mode="before")
    def _norm_recurrence(cls, v):
        if isinstance(v, bool):
            return "Yes" if v else "No"
        v = str(v).strip().lower()
        if v in {"yes", "true", "y", "1"}:
            return "Yes"
        if v in {"no", "false", "n", "0"}:
            return "No"
        raise ValueError("progression_or_recurrence must be Yes/No or boolean")

    @field_validator("vital_status", "gender", "race", mode="before")
    def _title_case(cls, v):
        return None if v is None else str(v).title()


@tool(args_schema=ClinicalRiskInput,
      )
def predict_clinical_risk_tool(**kwargs) -> str:
    """
    Use this tool **whenever** a user asks for an *individual* risk / prognosis
    based on patient demographics and diagnosis information.
    """
    return _predict_clinical_risk_internal(**kwargs)


@tool(return_direct=True)
def classify_image_tool(image_path: str) -> str:
    """
    Classify a blood-smear image as 'leukemic' or 'normal'.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = image_transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(cnn_model(tensor), dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, idx].item())
        return f"Image classified as '{class_names[idx]}' with {conf:.2%} confidence."
    except Exception as e:
        return f"Error processing image: {e}"


tools = [predict_clinical_risk_tool, classify_image_tool]

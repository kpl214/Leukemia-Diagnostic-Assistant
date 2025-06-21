# backend/langchain_agent.py
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tools import tools   # predict_clinical_risk + classify_image_tool

load_dotenv()

llm = ChatOpenAI(
    temperature=0.2,
    model=os.getenv("MODEL"),
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("BASE_URL"),
    
)

SYSTEM_MSG = """
You are an oncology assistant.

WHEN you receive an observation from the tool that contains
  "low_risk_probability"  (L)  and
  "high_risk_probability" (H):

• IF H > L (high-risk probability is greater)
    - Respond with BOTH numbers, formatted as:
        “There is H % chance the disease WILL recur or lead to mortality,
         and a L % chance it will NOT.”
    - If H ≥ 30 %, append a short note of caution and advise closer monitoring.

• IF L ≥ H (low-risk probability is greater or equal)
    - Respond ONLY with the low-risk number:
        “There is L % chance the disease will NOT recur or lead to mortality.”
    - Do NOT mention the high-risk percentage in this case.

Always finish with:  
“Model predictions are estimates and should be interpreted by a healthcare professional.”

For any other user question that does not involve an individual risk estimate, answer normally without calling the tool.

"""

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"prefix": SYSTEM_MSG},
)

def run_agent(prompt: str) -> str:
     return agent_executor.run(prompt)


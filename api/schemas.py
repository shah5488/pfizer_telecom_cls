from pydantic import BaseModel

class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int

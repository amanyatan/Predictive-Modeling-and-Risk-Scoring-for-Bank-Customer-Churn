from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from churn_pipeline import predict_customer_churn
from fastapi.responses import HTMLResponse

app = FastAPI(title="Bank Churn Predictor API")

class ChurnInput(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head><title>Bank Churn Predictor</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
            <h1>🏦 Bank Churn Predictor API is Running!</h1>
            <p>Vercel deployment successful.</p>
            <p>Go to <a href="/docs">/docs</a> to test the prediction API.</p>
        </body>
    </html>
    """

@app.post("/predict")
def predict(data: ChurnInput):
    try:
        input_data = data.dict()
        churn_prob = predict_customer_churn(input_data)
        return {
            "churn_probability": float(churn_prob),
            "high_risk": churn_prob > 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

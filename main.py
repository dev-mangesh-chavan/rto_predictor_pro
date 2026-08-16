from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import uvicorn
import logging

# Import our custom ML Engine from the src folder
from src.ml_predictor import RTOPredictor

# Setup Enterprise Logging with Timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI microservice
app = FastAPI(
    title="RTO Predictor Pro API",
    description="Enterprise API for real-time and batch E-commerce RTO risk prediction.",
    version="2.0" # Upgraded for 10k Data Model
)

# Configure CORS so our Streamlit UI can talk to this API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI Engine when the server starts
try:
    predictor = RTOPredictor()
    logger.info("✅ API Startup: ML Engine (v2.0) loaded successfully into memory.")
except Exception as e:
    logger.error(f"❌ API Startup Error: {e}")

# ---------------------------------------------------------
# Define the Request Format (Schema) for Single Order
# ---------------------------------------------------------
class OrderRequest(BaseModel):
    Cart_Value: float
    Return_Rate: float      # E.g. 0.20 for 20% past returns
    Pincode_Tier: int       # 1, 2, or 3
    Address_Quality: int    # Score from 1 to 10
    Is_COD: int             # 1 for Yes, 0 for No

# ---------------------------------------------------------
# 1. Endpoint: Health Check
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "Online", "message": "Welcome to RTO Predictor Pro API v2.0"}

# ---------------------------------------------------------
# 2. Endpoint: Predict Single Order (For Manual UI)
# ---------------------------------------------------------
@app.post("/predict/single")
async def predict_single(order: OrderRequest):
    """
    Takes a single order JSON, sends it to the predictor, and returns the risk analysis.
    """
    try:
        # Convert the incoming Pydantic model to a standard dictionary
        order_data = order.model_dump()
        
        # Ask the Chef (RTOPredictor) for the result
        result = predictor.predict_single(order_data)
        
        # Check if the predictor threw an error
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        return result

    except Exception as e:
        logger.error(f"Single Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ---------------------------------------------------------
# 3. Endpoint: Predict Batch via CSV (For Auto Batch Scan)
# ---------------------------------------------------------
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Takes a CSV file of multiple orders, analyzes all of them, and returns a detailed JSON list.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    try:
        # Read the uploaded CSV file directly into a Pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Failsafe: Check if the uploaded CSV is completely empty
        if df.empty:
            raise ValueError("The uploaded CSV file contains no data.")
            
        # Ask the Chef to process the entire batch
        results_df = predictor.predict_batch(df)
        
        # PRO-FIX: JSON cannot handle pandas NaN values. Replace them with None
        results_df = results_df.replace({np.nan: None})
        
        # Convert the resulting DataFrame back into a JSON-friendly dictionary
        results_json = results_df.to_dict(orient="records")
        
        return {
            "status": "success",
            "total_processed": len(results_json),
            "data": results_json
        }
        
    except pd.errors.EmptyDataError:
        logger.error("Batch Prediction failed: Uploaded CSV was empty.")
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty or corrupted.")
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Standard entry point for local server
if __name__ == "__main__":
    logger.info("🚀 Starting Enterprise FastAPI Server on port 8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
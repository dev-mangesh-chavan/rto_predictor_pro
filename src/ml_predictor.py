import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class RTOPredictor:
    """
    Industry-standard Prediction Class.
    Loads the trained assets once and provides methods for single and batch predictions.
    Now includes a Reason Generator to explain AI decisions!
    """
    def __init__(self):
        # Dynamically find the models folder (works on any OS)
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / 'models'
        
        self.model = None
        self.scaler = None
        self.processor = DataProcessor() # Connecting our new Security Guard
        self.feature_columns = ['Cart_Value', 'Return_Rate', 'Pincode_Tier', 'Address_Quality', 'Is_COD']
        
        # Load assets immediately when the server starts
        self._load_assets()

    def _load_assets(self):
        """Loads the saved ML model and scaler from disk."""
        try:
            model_path = self.models_dir / 'rto_rf_model.pkl'
            scaler_path = self.models_dir / 'feature_scaler.pkl'
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("✅ ML Model and Scaler loaded successfully into memory.")
        except Exception as e:
            logger.error(f"❌ Error loading assets. Did you run the training notebook? Details: {e}")

    def predict_single(self, order_data: dict) -> dict:
        """
        Takes a single order dictionary, scales it, returns risk analysis AND the logical reason.
        """
        if self.model is None or self.scaler is None:
            return {"error": "Model not loaded"}

        # Process and clean the data using our dedicated processor
        df = self.processor.preprocess_single(order_data)

        # Scale the data using our saved scaler
        scaled_data = self.scaler.transform(df)

        # Get probability of RTO (Class 1)
        risk_probability = self.model.predict_proba(scaled_data)[0][1]
        risk_percentage = round(risk_probability * 100, 2)

        # Extract features to write a sensible reason
        cart_val = order_data.get('Cart_Value', 0)
        ret_rate = order_data.get('Return_Rate', 0)
        is_cod = order_data.get('Is_COD', 0)

        # Business Logic / Bucketing & Reasoning
        if risk_percentage < 40:
            status = "Safe"
            color = "Green"
            action = "Auto-Approved. Proceed to packaging."
            
            # Smart Reason for Safe
            if is_cod == 0:
                reason = "Prepaid order (Zero financial risk)."
            elif ret_rate < 0.05:
                reason = "Excellent customer track record (Very low past returns)."
            else:
                reason = "Standard low-risk profile detected by AI."
                
        elif risk_percentage < 75:
            status = "Review"
            color = "Yellow"
            action = "High Risk. Send SMS for ₹50 advance payment."
            
            # Smart Reason for Review
            if is_cod == 1 and ret_rate > 0.15:
                reason = f"COD order with borderline return history ({int(ret_rate*100)}%)."
            elif cart_val > 5000:
                reason = "Unusually high cart value for a COD order. Needs verification."
            else:
                reason = "AI detected a moderate probability of cancellation."
                
        else:
            status = "Blocked"
            color = "Red"
            action = "Serial Returner. Order Auto-Cancelled."
            
            # Smart Reason for Blocked
            if ret_rate >= 0.50:
                reason = f"Serial Returner alert! Customer returns {int(ret_rate*100)}% of their orders."
            elif is_cod == 1 and cart_val > 10000:
                reason = "Extremely high value COD order from a risky profile."
            else:
                reason = "Multiple high-risk factors triggered the AI threshold."

        return {
            "risk_score": risk_percentage,
            "status": status,
            "color_code": color,
            "recommended_action": action,
            "ai_reason": reason # Passed securely to API
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame of multiple orders, predicts risk, and assigns reasons vectorized.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or Scaler is missing!")

        # Keep original data safe, work on a copy
        results_df = df.copy()
        
        # Clean and extract only the columns we need using processor
        X = self.processor.preprocess_batch(df)
        
        # Scale all rows at once
        scaled_data = self.scaler.transform(X)
        
        # Predict probabilities for the entire batch
        probabilities = self.model.predict_proba(scaled_data)[:, 1]
        
        results_df['Risk_Score_%'] = np.round(probabilities * 100, 2)
        
        # Apply Status logic using vectorized approach (Fast)
        cond_safe = results_df['Risk_Score_%'] < 40
        cond_review = (results_df['Risk_Score_%'] >= 40) & (results_df['Risk_Score_%'] < 75)
        cond_blocked = results_df['Risk_Score_%'] >= 75
        
        results_df['Status'] = np.select(
            [cond_safe, cond_review, cond_blocked], 
            ['Safe', 'Review', 'Blocked'], 
            default='Review'
        )

        # Apply Reason logic blazingly fast using Numpy
        r_serial = (results_df['Status'] == 'Blocked') & (results_df['Return_Rate'] >= 0.40)
        r_high_val_cod = (results_df['Status'] == 'Blocked') & (results_df['Cart_Value'] > 8000) & (results_df['Is_COD'] == 1)
        
        r_mod_ret = (results_df['Status'] == 'Review') & (results_df['Return_Rate'] > 0.15)
        r_rev_high_val = (results_df['Status'] == 'Review') & (results_df['Cart_Value'] > 5000)
        
        r_prepaid = (results_df['Status'] == 'Safe') & (results_df['Is_COD'] == 0)
        r_good_hist = (results_df['Status'] == 'Safe') & (results_df['Return_Rate'] < 0.05)

        # Map conditions to text explanations
        reason_texts = [
            "Serial Returner detected.",
            "High-value COD risk.",
            "Moderate return history.",
            "High cart value (Needs verification).",
            "Prepaid order (No financial risk).",
            "Excellent customer track record."
        ]

        # Default fallbacks if none of the above specific rules trigger
        default_reasons = np.select(
            [results_df['Status'] == 'Blocked', results_df['Status'] == 'Review', results_df['Status'] == 'Safe'],
            ["Multiple risk factors.", "Borderline profile.", "Standard safe profile."],
            default="AI Processed."
        )

        results_df['AI_Reason'] = np.select(
            [r_serial, r_high_val_cod, r_mod_ret, r_rev_high_val, r_prepaid, r_good_hist],
            reason_texts,
            default=default_reasons
        )
        
        return results_df
import pandas as pd
import logging

# Industry standard logging format with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Industry-standard Data Processing Class.
    Acts as a 'Security Guard' for the ML Model. Validates, cleans, and formats
    incoming raw API data (Single & Batch) before prediction.
    """
    def __init__(self):
        # 1. Exact sequence of columns the Random Forest was trained on.
        # Order is extremely important in Machine Learning!
        self.expected_columns = [
            'Cart_Value', 
            'Return_Rate', 
            'Pincode_Tier', 
            'Address_Quality', 
            'Is_COD'
        ]
        
        # 2. Strict Data Types to prevent API and Model crashes
        self.dtypes = {
            'Cart_Value': float,
            'Return_Rate': float,
            'Pincode_Tier': int,
            'Address_Quality': int,
            'Is_COD': int
        }

    def preprocess_single(self, raw_data: dict) -> pd.DataFrame:
        """
        Validates and converts a single dictionary into a clean, model-ready DataFrame.
        """
        logger.info("🛡️ Guard: Inspecting single order data...")
        
        # Convert dictionary to DataFrame (1 row)
        df = pd.DataFrame([raw_data])
        
        # Failsafe mechanism: Fill missing critical values gracefully
        if 'Return_Rate' not in df.columns or pd.isna(df['Return_Rate'].iloc[0]):
            df['Return_Rate'] = 0.05  # Default 5% safe assumption
            
        try:
            # Enforce exact columns and order
            clean_df = df[self.expected_columns].copy()
            
            # Enforce exact data types (Type Casting)
            clean_df = clean_df.astype(self.dtypes)
            
            return clean_df
            
        except KeyError as e:
            logger.error(f"❌ Missing required data fields: {e}")
            raise ValueError(f"Missing required fields for AI prediction: {e}")
        except Exception as e:
            logger.error(f"❌ Data format conversion failed: {e}")
            raise ValueError(f"Invalid data format received: {e}")

    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and cleans a bulk CSV upload dataframe.
        """
        logger.info(f"🛡️ Guard: Inspecting batch of {len(df)} orders...")
        
        # Ensure we don't accidentally modify the user's original uploaded file
        process_df = df.copy()
        
        # 1. Ensure all required columns exist in the uploaded CSV
        missing_cols = [col for col in self.expected_columns if col not in process_df.columns]
        if missing_cols:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing_cols}")
        
        # 2. Remove any corrupt/empty rows (Failsafe)
        process_df = process_df.dropna(subset=self.expected_columns)
        
        # 3. Extract exact columns in correct order
        clean_df = process_df[self.expected_columns].copy()
        
        # 4. Enforce strict data types to prevent model crash in batch
        clean_df = clean_df.astype(self.dtypes)
        
        logger.info("✅ Batch data validated and cleaned successfully.")
        return clean_df
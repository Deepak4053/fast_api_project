from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import io

app = FastAPI(
    title="California Housing Price Prediction API",
    version="1.0.0"
)

# Load model and feature names
model = joblib.load("california_housing_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# inputs for single prediction
class HousingData(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age")
    AveRooms: float = Field(..., description="Average rooms per household")
    AveBedrms: float = Field(..., description="Average bedrooms per household")
    Population: float = Field(..., description="Population")
    AveOccup: float = Field(..., description="Average household occupancy")
    Latitude: float = Field(
        ge=32,
        le=42,
        description="Latitude"
    )
    Longitude: float = Field(
        ge=-125,
        le=-114,
        description="Longitude"
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to the California Housing Price Prediction API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Single house prediction",
            "/predict/batch": "POST - Batch CSV prediction",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model": type(model).__name__
    }


# Single Prediction Endpoint
@app.post("/predict")
def predict(data: HousingData):
    try:
        input_data = pd.DataFrame([{
            "MedInc": data.MedInc,
            "HouseAge": data.HouseAge,
            "AveRooms": data.AveRooms,
            "AveBedrms": data.AveBedrms,
            "Population": data.Population,
            "AveOccup": data.AveOccup,
            "Latitude": data.Latitude,
            "Longitude": data.Longitude
        }])

        input_data = input_data[feature_names]
        prediction = model.predict(input_data)[0]
        price = prediction * 100000

        return {
            "predicted_price": round(price, 2),
            "formatted_price": f"${price:,.2f}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# Batch Prediction 
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        content = await file.read()
        df = pd.read_csv(
            io.StringIO(content.decode("utf-8"))
        )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty."
            )

        if len(df) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10,000 rows allowed."
            )

        missing_features = [
            col for col in feature_names
            if col not in df.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {', '.join(missing_features)}"
            )

        input_df = df[feature_names]
        predictions = model.predict(input_df)

        df["predicted_price"] = predictions * 100000
        df["formatted_price"] = df["predicted_price"].apply(
            lambda x: f"${x:,.2f}"
        )

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=predictions.csv"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )
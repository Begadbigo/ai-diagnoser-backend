from fastapi import FastAPI
from pydantic import BaseModel

# Import the functions you already built from your other file!
from rules_engine import check_vitals_for_alerts, determine_patient_status

# Create the FastAPI application instance
app = FastAPI()

# Define the shape of the data your API will expect to receive from the app
class VitalsInput(BaseModel):
    HR: int
    SpO2: int
    SystolicBP: int
    RR: int

# Define your first API endpoint at the URL path "/check-status/"
@app.post("/check-status/")
async def check_patient_status(vitals: VitalsInput):
    """
    This endpoint receives patient vitals, checks for alerts,
    determines a status, and returns the result as JSON.
    """
    # Convert the input data into a dictionary that our functions can use
    vitals_dict = vitals.dict()
    
    # Use the functions from your rules_engine.py file
    alerts = check_vitals_for_alerts(vitals_dict)
    status = determine_patient_status(alerts)
    
    # Return the results
    return {"patient_status": status, "alerts": alerts}

# A simple endpoint to check if the API is running
@app.get("/")
async def root():
    return {"message": "AI Diagnoser API is running!"}

"""
AI Diagnoser - Rules Engine (V2 - With Severity Logic)
This file contains the simple, rule-based algorithms for the application.
This version introduces severity levels for more medically logical status determination.
"""

# Step 1: Update the thresholds to include a 'severity' for each rule.
VITAL_THRESHOLDS = {
    'SpO2': {
        'low': {'value': 92, 'severity': 'critical'}, # Low SpO2 is always critical
    },
    'HR': { # Heart Rate
        'low': {'value': 50, 'severity': 'urgent'},
        'high': {'value': 120, 'severity': 'urgent'},
        'very_high': {'value': 140, 'severity': 'critical'}, # Add a more severe level
    },
    'SystolicBP': { # Systolic Blood Pressure
        'low': {'value': 90, 'severity': 'critical'}, # Low BP is very dangerous
        'high': {'value': 160, 'severity': 'urgent'},
    },
    # ... we can add more rules here
}

def check_vitals_for_alerts(current_vitals: dict) -> list[dict]:
    """
    Checks vitals and returns a list of alert objects, each with a message and severity.
    """
    alerts = []

    # Check SpO2
    if 'SpO2' in current_vitals and current_vitals['SpO2'] < VITAL_THRESHOLDS['SpO2']['low']['value']:
        alerts.append({'message': 'Hypoxia (Low SpO2)', 'severity': VITAL_THRESHOLDS['SpO2']['low']['severity']})
        
    # Check HR
    if 'HR' in current_vitals:
        if current_vitals['HR'] > VITAL_THRESHOLDS['HR']['very_high']['value']:
            alerts.append({'message': 'Extreme Tachycardia', 'severity': VITAL_THRESHOLDS['HR']['very_high']['severity']})
        elif current_vitals['HR'] > VITAL_THRESHOLDS['HR']['high']['value']:
            alerts.append({'message': 'Tachycardia (High Heart Rate)', 'severity': VITAL_THRESHOLDS['HR']['high']['severity']})
        elif current_vitals['HR'] < VITAL_THRESHOLDS['HR']['low']['value']:
            alerts.append({'message': 'Bradycardia (Low Heart Rate)', 'severity': VITAL_THRESHOLDS['HR']['low']['severity']})
            
    # Check Systolic BP
    if 'SystolicBP' in current_vitals and current_vitals['SystolicBP'] < VITAL_THRESHOLDS['SystolicBP']['low']['value']:
        alerts.append({'message': 'Hypotension (Low Blood Pressure)', 'severity': VITAL_THRESHOLDS['SystolicBP']['low']['severity']})

    return alerts

def determine_patient_status(alerts: list[dict]) -> str:
    """
    Determines patient status based on the HIGHEST severity alert found.
    """
    severities = [alert['severity'] for alert in alerts]
    
    if 'critical' in severities:
        return "Critical"
    elif 'urgent' in severities:
        return "Urgent"
    else:
        return "Stable"

# --- Updated Testing Block ---
if __name__ == "__main__":
    print("--- Testing the V2 Rules Engine with Severity Logic ---")

    # Test Case 1: Patient with ONE critical alert (should be "Critical")
    patient_1 = {'SpO2': 88}
    print(f"\nTesting with Patient 1 vitals: {patient_1}")
    alerts_1 = check_vitals_for_alerts(patient_1)
    status_1 = determine_patient_status(alerts_1)
    print(f"--> Alerts found: {alerts_1}")
    print(f"--> Determined Status: {status_1}")
    assert status_1 == "Critical"

    # Test Case 2: Patient with TWO urgent alerts (should still be "Urgent")
    patient_2 = {'HR': 125, 'SystolicBP': 170}
    print(f"\nTesting with Patient 2 vitals: {patient_2}")
    alerts_2 = check_vitals_for_alerts(patient_2)
    status_2 = determine_patient_status(alerts_2)
    print(f"--> Alerts found: {alerts_2}")
    print(f"--> Determined Status: {status_2}")
    assert status_2 == "Urgent"

    # Test Case 3: A stable patient
    patient_3 = {'HR': 80, 'SpO2': 98}
    print(f"\nTesting with Patient 3 vitals: {patient_3}")
    alerts_3 = check_vitals_for_alerts(patient_3)
    status_3 = determine_patient_status(alerts_3)
    print(f"--> Alerts found: {alerts_3}")
    print(f"--> Determined Status: {status_3}")
    assert status_3 == "Stable"

    print("\n--- All tests passed successfully! ---")

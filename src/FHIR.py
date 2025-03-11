import requests
from datetime import datetime
import random

print("Fetching FIHR Data")
patients_url = "https://fhirsandbox.healthit.gov/open/r4/fhir/Patient?_format=json"
patients_data = requests.get(patients_url).json()

patients_dict = {}

today = datetime.today()

for entry in patients_data.get("entry", []):
    resource = entry["resource"]
    patient_id = resource["id"]

    name_info = resource.get("name", [{}])[0]
    given_name = " ".join(name_info.get("given", []))
    family_name = name_info.get("family", "")
    full_name = f"{given_name} {family_name}".strip()

    birth_date = resource.get("birthDate")
    age = None
    if birth_date:
        birth_date_dt = datetime.strptime(birth_date, "%Y-%m-%d")
        age = today.year - birth_date_dt.year - ((today.month, today.day) < (birth_date_dt.month, birth_date_dt.day))

    ethnicity = "Unknown"
    for ext in resource.get("extension", []):
        if ext.get("url") == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity":
            ethnicity = ext.get("extension", [{}])[1].get("valueString", "Unknown")

    patients_dict[patient_id] = {
        "name": full_name,
        "age": age,
        "ethnicity": ethnicity,
        "medications": set(),
        "careplans": set(),
        "body_weight": None,
        "body_length": None,
    }

    # Here we fetch medic data (Catalog, not patient-specific) i.e. medicines in the catalogue
    med_url = "https://fhirsandbox.healthit.gov/open/r4/fhir/Medication?_format=json"
    med_data = requests.get(med_url).json()
    if med_data.get("entry"):
        for med_entry in med_data["entry"]:
            med_coding = med_entry["resource"].get("code", {}).get("coding", [])
            for code in med_coding:
                patients_dict[patient_id]["medications"].add(code.get('display', 'Unknown'))

    # This fetches from CARE-PLANS (Patient-specific)
    careplan_url = f"https://fhirsandbox.healthit.gov/open/r4/fhir/CarePlan?subject=Patient/{patient_id}&_format=json"
    careplan_data = requests.get(careplan_url).json()
    if careplan_data.get("entry"):
        for cp_entry in careplan_data["entry"]:
            cp_coding_list = cp_entry["resource"].get("category", [])
            for cp_coding in cp_coding_list:
                for code in cp_coding.get("coding", []):
                    patients_dict[patient_id]["careplans"].add(code.get('display', 'Unknown'))

    # this fetches from OBSERVATIONS (Weight and Length)
    obs_url = f"https://fhirsandbox.healthit.gov/open/r4/fhir/Observation?subject=Patient/{patient_id}&_format=json"
    obs_data = requests.get(obs_url).json()
    if obs_data.get("entry"):
        for obs_entry in obs_data["entry"]:
            obs = obs_entry["resource"]
            code_text = obs.get("code", {}).get("text", "").lower()
            value = obs.get("valueQuantity", {}).get("value")
            unit = obs.get("valueQuantity", {}).get("unit", "")

            if "body weight" in code_text:
                patients_dict[patient_id]["body_weight"] = f"{value} {unit}"
            if "body height" in code_text or "body length" in code_text:
                patients_dict[patient_id]["body_length"] = f"{value} {unit}"

'''
for patient_id, details in patients_dict.items():
    print(f"\nPatient: {details['name']} (ID: {patient_id}), Age: {details['age']}, Ethnicity: {details['ethnicity']}")
    print("Medications:", ", ".join(details['medications']) or "None")
    print("CarePlans:", ", ".join(details['careplans']) or "None")
    print(f"Body Weight: {details['body_weight'] or 'Unknown'}")
    print(f"Body Length: {details['body_length'] or 'Unknown'}")
'''

patient_id = random.choice(list(patients_dict.keys()))  

def get_random_patient():
    if not patients_dict:
        return None  
    
    details = patients_dict[patient_id]

    patient_info = {
        "type": "patient_info",
        "name": details["name"],
        "id": patient_id,
        "age": details["age"],
        "medications": list(details["medications"]) or ["None"],
        "careplans": list(details["careplans"]) or ["None"],
        "contact": "benjaminp1997@hotmail.com"
    }

    return patient_info
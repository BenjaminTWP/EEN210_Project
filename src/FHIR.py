import requests

patients_url = "https://fhirsandbox.healthit.gov/open/r4/fhir/Patient?_format=json"
patients_data = requests.get(patients_url).json()

patients_dict = {}

for entry in patients_data.get("entry", []):
    patient_id = entry["resource"]["id"]
    name_info = entry["resource"].get("name", [{}])[0]
    given_name = " ".join(name_info.get("given", []))
    family_name = name_info.get("family", "")
    full_name = f"{given_name} {family_name}".strip()

    patients_dict[patient_id] = {"name": full_name, "medications": set(), "careplans": set()}

    # Fetch Medications
    med_url = f"https://fhirsandbox.healthit.gov/open/r4/fhir/MedicationRequest?subject=Patient/{patient_id}&_format=json"
    med_data = requests.get(med_url).json()

    if med_data.get("entry"):
        for med_entry in med_data["entry"]:
            med_coding = med_entry["resource"].get("medicationCodeableConcept", {}).get("coding", [])
            for code in med_coding:
                patients_dict[patient_id]["medications"].add(code.get('display', 'Unknown'))

    # Fetch CarePlans
    careplan_url = f"https://fhirsandbox.healthit.gov/open/r4/fhir/CarePlan?subject=Patient/{patient_id}&_format=json"
    careplan_data = requests.get(careplan_url).json()

    if careplan_data.get("entry"):
        for cp_entry in careplan_data["entry"]:
            cp_coding_list = cp_entry["resource"].get("category", [])
            for cp_coding in cp_coding_list:
                for code in cp_coding.get("coding", []):
                    patients_dict[patient_id]["careplans"].add(code.get('display', 'Unknown'))

# Print the dictionary in a readable format
for patient_id, details in patients_dict.items():
    print(f"\nPatient: {details['name']} (ID: {patient_id})")
    print("Medications:", ", ".join(details['medications']) or "None")
    print("CarePlans:", ", ".join(details['careplans']) or "None")

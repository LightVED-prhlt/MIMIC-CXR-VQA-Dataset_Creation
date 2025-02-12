import os
import re

def extract_indication(images_path, base_path):
    """
    Extracts the indication or clinical history from a text file corresponding to a medical image study.
    
    Args:
        images_path (str): Path to the image file, used to derive patient and study IDs.
        base_path (str): Base directory where patient study text reports are stored.
    
    Returns:
        str: Extracted indication text if found, or a relevant message if not.
    """
    path_parts = images_path.split('/')
    if len(path_parts) >= 3:
        patient_id = path_parts[1]
        study_id = path_parts[2]
    else:
        return "Unknown"

    formatted_path = os.path.join(base_path, patient_id[:3], patient_id, f"{study_id}.txt")

    try:
        with open(formatted_path, "r", encoding="utf-8") as file:
            text = file.read()

        patterns = [
            r"INDICATION:\s*(.*?)(?:\n\s*\n|$)",
            r"HISTORY:\s*(.*?)(?:\n\s*\n|$)",
            r"CLINICAL INFORMATION:\s*(.*?)(?:\n\s*\n|$)",
            r"REASON FOR EXAMINATION:\s*(.*?)(?:\n\s*\n|$)",
            r"CLINICAL HISTORY  History:\s*(.*?)(?:\n\s*\n|$)",
            r"REASON FOR EXAM:\s*(.*?)(?:\n\s*\n|$)",
            r"REASON FOR THE EXAM:\s*(.*?)(?:\n\s*\n|$)",
            r"(?i)Indication[:;]\s*(.*?)(?:\n\s*\n|$)",
            r"INDICATION\s*(.*?)(?:\n\s*\n|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return f'{" ".join(match.group(1).split())}'

        return "No indication found"
    
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {str(e)}"

import ollama
import os
import pandas as pd
from tqdm import tqdm
import argparse
from extract_indication import extract_indication
from GenerarPreguntas import QuestionGenerator

# Set up the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--reports_split", type=str, default="/home/Data/NEW/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/train.metadata.csv", help="The split of the reports to generate questions and answers for (train, validate, test)")
parser.add_argument("--mimic_reports", type=str, default="/home/Data/NEW/mimic-cxr/2.0.0/mimic-cxr-reports/", help="Base path for MIMIC reports")
parser.add_argument("--chexpert_labels", type=str, default="mimic-cxr-2.0.0-chexpert.csv", help="Path for csv containing the chexpert labels")
parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing the reports")
parser.add_argument("--end_index", type=int, default=152173, help="Ending index for processing the reports")

args = parser.parse_args()

reports_path = args.reports_split
output_csv = "generated_questions_answers_{}-{}.csv".format(args.start_index, args.end_index - 1)
question_gen = QuestionGenerator(args.chexpert_labels)

data = pd.read_csv(reports_path)

def generate_question_answer(report, question, indication):
    """
    Generates a question-answer pair based on a radiology report, a given question, and an indication.
    """
    prompt_template = f"""
        indication: {indication}
        radiograph: {report}
        question: {question}
    """
    
    response = ollama.generate(model="custom-llama3.1:8b", prompt=prompt_template)
    
    try:
        qa_pair = eval(response["response"])
    except:
        qa_pair = {"question": question, "answer": "error"}
    
    return qa_pair

# Extract indications
data["indication"] = data["images"].apply(lambda x: extract_indication(x, args.mimic_reports))
data.to_csv("updated_reports.csv", index=False)
print("Updated dataset saved to updated_reports.csv")

columns = ["index", "patient_id", "study_id", "question", "answer", "question_label", "question_value", "image_path", "view_position"]
if not os.path.exists(output_csv):
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

buffer = []
save_interval = 100

# Slice data based on provided indices
start_index = args.start_index
end_index = args.end_index if args.end_index is not None else len(data)
data = data.iloc[start_index:end_index]

print(f"Processing data from index {start_index} to {end_index - 1}")

for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
    actual_index = start_index + index
    
    text = row['text']
    images_path = row['images']
    view_position = row['ViewPosition']
    indication = row['indication']
    
    path_parts = images_path.split('/')
    if len(path_parts) >= 3:
        patient_id = path_parts[1]  
        study_id = path_parts[2]  
    else:
        patient_id, study_id = "Unknown", "Unknown"

    labels_dict = question_gen.get_labels_dict(patient_id[1:], study_id[1:])
    labels_dict["heart"] = 1.0 if "heart" in text.lower() else -100.0
    
    questions = question_gen.generate_questions(labels_dict)

    for label, value, question in questions:
        qa_pair = generate_question_answer(text, question, indication)

        row_data = {
            "index": actual_index,
            "patient_id": patient_id,
            "study_id": study_id,
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
            "question_label": label,
            "question_value": value,
            "image_path": images_path,
            "view_position": view_position
        }
        buffer.append(row_data)

    if actual_index % save_interval == 0 and actual_index != 0:
        pd.DataFrame(buffer).to_csv(output_csv, mode='a', header=False, index=False)
        buffer = []
        print(f"Saved {save_interval} rows to {output_csv}")

if buffer:
    pd.DataFrame(buffer).to_csv(output_csv, mode='a', header=False, index=False)
    print(f"Saved final {len(buffer)} rows to {output_csv}")

print(f"Question-answer pairs saved to {output_csv}")

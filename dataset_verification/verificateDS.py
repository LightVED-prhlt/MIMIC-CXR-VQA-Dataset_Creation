import ollama
import os
import pandas as pd
from tqdm import tqdm
import argparse
import re
import json

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--reports_file", type=str, default="../updated_reports.csv", help="Path to the updated reports CSV file")
parser.add_argument("--qa_file", type=str, default="../generated_questions_answers_train_all.csv", help="Path to the QA pairs CSV file")
parser.add_argument("--num_reports", type=int, default=3000, help="Number of reports to evaluate")
parser.add_argument("--output_file", type=str, default="evaluated_qa_pairs2.csv", help="Path to save the output CSV")
args = parser.parse_args()

# Load data
reports_df = pd.read_csv(args.reports_file)
qa_df = pd.read_csv(args.qa_file)

# # Select the first N reports to evaluate
# reports_df = reports_df.iloc[:args.num_reports]

# Select N reports randomly
reports_df = reports_df.sample(n=args.num_reports, random_state=42)

def extract_final_dict(response_text):
    """
    Extracts the final dictionary from the response text.
    """
    # print(response_text)
    matches = re.findall(r'\{.*?\}', response_text, re.DOTALL)
    extracted_dict = {"correctness": "Error", "consistency_with_report": "Error", "answer_completeness": "Error", "clinical_relevance": "Error", "reasoning": "Error"}
    for match in reversed(matches):  # Take the last match, as it's most likely the final dictionary
        try:
            # print(match)
            extracted_dict = eval(match)  # Convert single quotes to double for JSON parsing
            extracted_dict["reasoning"] = response_text if extracted_dict.get("correctness") == "Incorrect" else "-"
            # if isinstance(extracted_dict, dict):
            #     print("Extracted Dictionary:", extracted_dict)  # Debugging statement
            #     return extracted_dict
        except:
            pass
    return extracted_dict

def evaluate_qa_pair(report, question, answer, indication):
    """
    Evaluates a given question-answer pair based on the radiology report.
    """
    prompt_template = f"""
    You are an assistant that, given a medical report about a chest radiograph, an indication for the imaging study, and a related question, will evaluate the given question-answer pair and provide structured annotations. Your task is to assess the correctness of the provided answer based strictly on the radiographic findings in the report, determine the consistency of the answer with the information in the report, evaluate the completeness of the answer (i.e., whether it includes all relevant details present in the report), and assess the clinical relevance of the answer in the context of the question. Base your judgment solely on the information present in the radiographic report, without making inferences or assumptions beyond what is explicitly described. Maintain an objective and neutral tone. At the end of your evaluation, provide a structured output in the following format: {{"correctness": "Correct" or "Incorrect", "consistency_with_report": "Fully Consistent" / "Partially Consistent" / "Inconsistent", "answer_completeness": "Complete" / "Partially Complete" / "Incomplete", "clinical_relevance": "Essential" / "Non-essential but Correct" / "Incorrect and Misleading"}}.
    
    indication: {indication}
    radiograph: {report}
    question: {question}
    answer: {answer}
    """
    
    response = ollama.generate(model="deepseek-r1:8b", prompt=prompt_template)
    
    return extract_final_dict(response["response"])

# Prepare output
output_data = []

# Process only reports from index 1395 to 1500
partial_reports_df = reports_df.reset_index(drop=True).iloc[1395:1500]

for i, (_, report_row) in enumerate(tqdm(partial_reports_df.iterrows(), total=len(partial_reports_df), desc="Processing Reports")):
    text = report_row['text']
    indication = report_row['indication']
    images_path = report_row['images']
    
    # Extract patient_id and study_id
    path_parts = images_path.split('/')
    patient_id = path_parts[1] if len(path_parts) >= 3 else "Unknown"
    study_id = path_parts[2] if len(path_parts) >= 3 else "Unknown"
    
    # Filter QA pairs that match the patient and study ID
    matching_qa_pairs = qa_df[(qa_df['patient_id'] == patient_id) & (qa_df['study_id'] == study_id)]
    
    for _, qa_row in matching_qa_pairs.iterrows():
        question = qa_row['question']
        answer = qa_row['answer']
        
        # Evaluate the QA pair
        # print(text)
        evaluation = evaluate_qa_pair(text, question, answer, indication)
        
        output_data.append({
            "patient_id": patient_id,
            "study_id": study_id,
            "question": question,
            "answer": answer,
            "correctness": evaluation["correctness"],
            "consistency_with_report": evaluation["consistency_with_report"],
            "answer_completeness": evaluation["answer_completeness"],
            "clinical_relevance": evaluation["clinical_relevance"],
            "reasoning": evaluation["reasoning"]
        })
        # print(text)
    
    # Save to CSV every 2 reports
    if (i + 1) % 2 == 0 or i == len(reports_df) - 1:
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(args.output_file, index=False)
        # print(f"Intermediate results saved to {args.output_file}")

print(f"Evaluation complete. Final results saved to {args.output_file}")

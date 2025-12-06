import pandas as pd
import random
import numpy as np

class QuestionGenerator:
    def __init__(self, csv_path):
        """
        Initializes the QuestionGenerator by loading the dataset from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing label data.
        """
        self.data = pd.read_csv(csv_path)
        self.question_dict = {
            "Atelectasis": ["Is Atelectasis observed?", "Can Atelectasis be identified?", "Are there signs of Atelectasis?", "What evidence of Atelectasis is present?", "Does the patient have Atelectasis?", "Are there indications of Atelectasis?"],
            "Cardiomegaly": ["Is Cardiomegaly present?", "Can Cardiomegaly be detected?", "Are there signs of Cardiomegaly?", "What suggests Cardiomegaly in the findings?", "Is there evidence of Cardiomegaly?", "Does the image show Cardiomegaly?"],
            "Consolidation": ["Is Consolidation observed?", "Can Consolidation be identified?", "Are there signs of Consolidation?", "What evidence of Consolidation is visible?", "Does the image indicate Consolidation?", "Are there indications of Consolidation?"],
            "Edema": ["Is Edema observed?", "Can Edema be detected?", "Are there signs of Edema?", "What evidence of Edema is present?", "Does the image suggest Edema?", "Are there indications of Edema?"],
            "Enlarged Cardiomediastinum": ["Is the Cardiomedistinum enlarged?", "Can an enlarged Cardiomedistinum be detected?", "Are there signs of an enlarged Cardiomedistinum?", "What suggests an enlarged Cardiomedistinum?", "Is the Cardiomedistinum abnormal in size?", "Are there indications of an enlarged Cardiomedistinum?"],
            "Fracture": ["Is a Fracture visible?", "Can a Fracture be identified?", "Are there signs of a Fracture?", "What evidence of a Fracture is present?", "Does the image indicate a Fracture?", "Are there indications of a Fracture?"],
            "Lung Lesion": ["Is a Lung Lesion observed?", "Can a Lung Lesion be detected?", "Are there signs of a Lung Lesion?", "What suggests a Lung Lesion in the findings?", "Is there evidence of a Lung Lesion?", "Does the image show a Lung Lesion?"],
            "Lung Opacity": ["Is Lung Opacity present?", "Can Lung Opacity be identified?", "Are there signs of Lung Opacity?", "What evidence of Lung Opacity is visible?", "Does the image indicate Lung Opacity?", "Are there indications of Lung Opacity?"],
            "No Finding": ["Are there any findings?", "Is there any findings in the image?", "Can any abnormalities be detected?", "Are there indications of findings in the image?","Are there any abnormalities present?","Do you see any findings in the image?"],
            "Pleural Effusion": ["Is Pleural Effusion present?", "Can Pleural Effusion be detected?", "Are there signs of Pleural Effusion?", "What evidence of Pleural Effusion is visible?", "Does the image suggest Pleural Effusion?", "Are there indications of Pleural Effusion?"],
            "Pleural Other": ["Is there any Pleural abnormality?", "Can Pleural findings be identified?", "Are there signs of Pleural conditions?", "What Pleural abnormalities are present?", "Are there unusual Pleural findings?", "Does the image suggest any Pleural conditions?"],
            "Pneumonia": ["Is Pneumonia observed?", "Can Pneumonia be detected?", "Are there signs of Pneumonia?", "What evidence of Pneumonia is present?", "Does the image indicate Pneumonia?", "Are there indications of Pneumonia?"],
            "Pneumothorax": ["Is Pneumothorax present?", "Can Pneumothorax be detected?", "Are there signs of Pneumothorax?", "What evidence of Pneumothorax is visible?", "Does the image suggest Pneumothorax?", "Are there indications of Pneumothorax?"],
            "Support Devices": ["Which Support Devices are found?", "What types of support devices are present?", "Which support devices can be identified?", "Are there any support devices observed?", "What support devices are detected?", "Can any support devices be found?"],
            "heart":["Is the heart size normal?","Does the heart appear enlarged?","How is the size of the heart?","Is the heart size within the expected range?","Does the radiograph suggest a normal heart size?","Is there any indication of an enlarged heart?"]
        }

    def get_labels_dict(self, subject_id, study_id):
        """
        Retrieves the labels for a given subject and study ID from the dataset.
        
        Args:
            subject_id (int): The ID of the subject.
            study_id (int): The ID of the study.
        
        Returns:
            dict or None: A dictionary of label names and their corresponding values,
            or None if the subject and study ID are not found in the dataset.
        """
        subject_id = int(subject_id)
        study_id = int(study_id)

        row = self.data[(self.data["subject_id"] == subject_id) & (self.data["study_id"] == study_id)]
        
        if not row.empty:
            return row.iloc[0].drop(["subject_id", "study_id"]).to_dict()
        return None


    def generate_questions(self, labels_dict):
        """
        Generates a list of questions based on the provided label dictionary.
        
        Args:
            labels_dict (dict): A dictionary where keys are label names and values indicate presence (0, 1, -1, or NaN).
        
        Returns:
            list: A list of tuples where each tuple contains (label, value, question).
        """
        question_list = []
        count_ones = sum(1 for value in labels_dict.values() if value == 1)
        
        # Extract labels with NaN values except "No Finding"
        nan_labels = [label for label, value in labels_dict.items() if label != 'No Finding' and isinstance(value, float) and np.isnan(value)]

        # Generate questions based on labels with defined values (0, -1, 1)
        for label, value in labels_dict.items():
            if value in [0, -1, 1] and label in self.question_dict:
                for question in self.question_dict[label]:
                    question_list.append((label, value, question))

        # Add 4 random questions per each "1" found
        selected_questions = set()
        for _ in range(4 * count_ones):
            if nan_labels:  # Ensure there are NaN labels available
                random_label = random.choice(nan_labels)
                if random_label in self.question_dict and self.question_dict[random_label]:
                    possible_questions = list(set(self.question_dict[random_label]) - selected_questions)
                    if possible_questions:
                        chosen_question = random.choice(possible_questions)
                        selected_questions.add(chosen_question)
                        question_list.append((random_label, float('nan'), chosen_question))

        return question_list
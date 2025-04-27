# Depression Prediction Model

This repository contains a machine learning model trained to predict depression in students based on academic and personal factors. The model uses a Random Forest Classifier and is built using Python's scikit-learn library.

## Features

- **Input Features**:
  - Academic Pressure (0-5 scale)
  - Study Satisfaction (0-5 scale)
  - History of Suicidal Thoughts (Yes/No)
  - Family History of Mental Illness (Yes/No)

- **Target Prediction**:
  - Binary Depression Prediction (0 = No, 1 = Yes)

## Prerequisites

- Python 3.x
- pandas
- scikit-learn
- numpy

Install dependencies:
```bash
pip install pandas scikit-learn numpy
```

## Dataset
The model uses a student depression dataset containing:
- Academic Pressure
- Study Satisfaction
- Suicidal Thoughts history
- Family mental health history
- Depression status (target variable)

*Replace `./student-depression-dataset/student_depression_dataset.csv` with your dataset path in the code.*

## Model Training
- **Algorithm**: Random Forest Classifier
- **Preprocessing**: One-hot encoding for categorical features
- **Train-Test Split**: 80% training, 20% testing
- **Accuracy**: **80.16%** (as per current dataset)

## Usage
1. Run the script:
```python
python depression_predictor.py
```

2. Enter values when prompted:
```bash
Enter Academic Pressure (0-5): 4
Enter Study Satisfaction (0-5): 2
Have you ever had suicidal thoughts? (0 for No, 1 for Yes): 1
Family History of Mental Illness? (0 for No, 1 for Yes): 1
```

3. Example prediction output:
```bash
Depression Prediction: 1
```

## Limitations
- Simplified binary classification
- Limited to 4 input features
- Accuracy depends on dataset quality (currently 80.16%)
- Not a substitute for professional diagnosis

## Ethical Considerations
This model is intended for educational purposes only. Always consult mental health professionals for clinical assessments.

## License
[MIT License](LICENSE)

# AIT716-Stress-Detection
AIT-716 final project implementing DistilBERT, TF-IDF logistic regression, and evaluation pipelines for stress detection in the Dreaddit Reddit dataset.
# Deep Learning-Based Detection of Population-Level Stress Signals in Social Media

This repository contains the code and supporting materials for the AIT-716 Advanced Artificial Intelligence final project at Capitol Technology University.

The project evaluates whether deep learning can identify stress-related language in Reddit posts and serve as a foundation for future population-level analysis of stress-related discourse. The system is designed as a research classifier and is **not intended for clinical diagnosis or individual mental-health assessment**.

## Project Overview

The project uses the **Dreaddit** dataset, which contains Reddit text labeled as stressful or non-stressful.

The primary model is:

* DistilBERT (`distilbert-base-uncased`)

Two reference models are also used:

* TF-IDF Logistic Regression
* Majority-Class DummyClassifier

The project compares the contextual language representations learned by DistilBERT with a traditional lexical baseline.

## Repository Structure

```text
AIT716-Stress-Detection/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── AIT716_Milestone_B_Initial_Notebook.ipynb
│   └── AIT716_Milestone_C_Prototype.ipynb
│
└── reports/
    └── AIT716_Final_Project_Report.pdf
```

### File Descriptions

**AIT716_Milestone_B_Initial_Notebook.ipynb**

Contains the initial dataset analysis and project design work, including:

* Dataset inspection
* Class distribution analysis
* Missing-value checks
* Duplicate-text detection
* Train/test overlap detection
* Initial preprocessing design
* Evaluation strategy

**AIT716_Milestone_C_Prototype.ipynb**

Contains the final end-to-end implementation, including:

* Dataset loading and cleaning
* Duplicate and data-leakage removal
* Conservative text preprocessing
* Train/validation splitting
* TF-IDF logistic-regression baseline
* Majority-class baseline
* DistilBERT tokenization and model configuration
* Transformer fine-tuning
* Validation metrics
* Held-out test evaluation
* Confusion matrix
* ROC-AUC evaluation
* Learning curves
* Error analysis
* Model export

**AIT716_Final_Project_Report.pdf**

Contains the final technical report describing the motivation, literature review, methodology, results, discussion, ethical considerations, limitations, and potential future research directions.

## Environment and Dependencies

The project was developed in Python and executed primarily in Google Colab using GPU acceleration.

Main dependencies include:

* Python 3.x
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* scikit-learn
* pandas
* NumPy
* Matplotlib
* Accelerate

Dependencies can be installed with:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Obtain the Dreaddit training and test datasets.
2. Open the Milestone C prototype notebook in Google Colab or Jupyter Notebook.
3. Update the dataset file paths if necessary.
4. Run the notebook cells sequentially.
5. GPU acceleration is recommended for DistilBERT fine-tuning.
6. The notebook will train the baseline models and DistilBERT and generate evaluation metrics and visualizations.

## Model Configuration

The primary DistilBERT configuration includes:

* Base model: `distilbert-base-uncased`
* Maximum sequence length: 256 tokens
* Training batch size: 16
* Evaluation batch size: 32
* Learning rate: 2 × 10^-5
* Epochs: 3
* Weight decay: 0.01
* Warmup ratio: 0.10
* Gradient clipping: 1.0
* Random seed: 42
* Model selection metric: Validation F1

## Final Test Results

DistilBERT achieved the following performance on the held-out test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8084 |
| Precision | 0.7915 |
| Recall    | 0.8537 |
| F1 Score  | 0.8214 |
| ROC-AUC   | 0.8783 |

The held-out test confusion matrix contained:

* 263 true negatives
* 83 false positives
* 54 false negatives
* 315 true positives

The transformer also outperformed the TF-IDF logistic-regression baseline on the common validation set.

## Research Scope

The model detects linguistic patterns associated with stress-related language. Its predictions should not be interpreted as clinical diagnoses or assessments of individual mental-health conditions.

Future development could extend the classifier into an aggregate social-signal system that separately measures:

* Intensity of stress-related discourse
* Prevalence across users or communities
* Persistence over time
* Amplification or disproportionate repetition

These dimensions could help distinguish broadly distributed social stress from concentrated or highly amplified reactions.

## Ethical Considerations

Important limitations and risks include:

* Privacy concerns involving social-media data
* Lack of demographic representativeness in Reddit data
* Concept drift because the Dreaddit dataset is historical
* False positives caused by emotionally difficult subject matter
* Potential misuse for individual surveillance or profiling
* Overinterpretation of amplified online discourse as representative of a larger population

The system is intended for research and aggregate analysis with appropriate human oversight.

## Author

Matias Romero
AIT-716 Advanced Artificial Intelligence
Capitol Technology University
2026

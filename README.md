# BBC Text Analysis

A Jupyter Notebook project that explores BBC news articles and predicts their news category using text mining and machine learning. Developed as a final assignment, it covers text preprocessing, keyword extraction, association rule mining, and classification with Bag of Words and TF-IDF features.

## Dataset

[`bbc-text.csv`](bbc-text.csv) contains **2,225 articles** with two columns:

| Column | Description |
| --- | --- |
| `category` | The article's news category, used as the classification target. |
| `text` | The article text, used for keyword extraction and model features. |

| Category | Articles |
| --- | ---: |
| sport | 511 |
| business | 510 |
| politics | 417 |
| tech | 401 |
| entertainment | 386 |
| **Total** | **2,225** |

The included CSV matches the notebook's expected filename, column names, saved row count, category totals, and first five displayed articles in order. The full first article also matches the notebook's saved output exactly. The file is preserved unchanged, including 99 duplicate rows, and has no missing category or text values.

The original [`stopwords.txt`](stopwords.txt) is also included unchanged. It contains 429 nonblank lines and 423 unique stop words. These two files are the notebook's only input files. `bbc-text_export.csv` is generated during execution and is excluded from Git.

## Analysis workflow

1. **Preprocess the text:** convert articles to lowercase and use regular expressions to remove tags, digits, and non-word characters. Load a custom stop-word list from `stopwords.txt` for feature extraction.
2. **Build text features:** create a Bag of Words matrix with up to 5,000 features, excluding terms found in more than 15% of articles, then transform it into TF-IDF weights.
3. **Extract keywords:** identify the five highest-weighted TF-IDF terms per article and export the enriched data to `bbc-text_export.csv`.
4. **Explore each category:** display keyword frequency charts and word clouds, then use Apriori and association rules to examine keywords that occur together, with support, confidence, and lift measures.
5. **Classify articles:** compare Logistic Regression, Decision Tree, and Random Forest classifiers using both feature representations, 70/30 train/test splits, 20-fold cross-validation, and grid search.

## Verified results

These accuracy scores were reproduced by executing all **98 nonempty code cells** in a fresh Python 3.13.0 environment on macOS with the pinned requirements. Verification also checked all **20 figures** and the **2,225-row keyword export**. The notebook contains the refreshed outputs. Scores are rounded to one decimal percentage point.

| Classifier | Features | Best cross-validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Logistic Regression | Bag of Words | 97.5% | 97.9% |
| Logistic Regression | TF-IDF | 97.4% | 98.1% |
| Decision Tree | Bag of Words | 83.9% | 85.0% |
| Decision Tree | TF-IDF | 84.3% | 85.6% |
| Random Forest | Bag of Words | 95.7% | 97.3% |
| Random Forest | TF-IDF | 96.3% | 97.2% |

Logistic Regression achieved the highest test accuracy in this run, at **98.1% with TF-IDF features**. The updated libraries, corrected Random Forest grid, and fixed random seeds mean some results differ from the original assignment outputs. See the evaluation limitations below when interpreting these results.

## Open the notebook

On macOS or Linux, clone the repository and launch Jupyter from its root directory so the relative dataset path resolves correctly:

```bash
git clone https://github.com/Goh-Wei-Chong/BBC-Text-Analysis.git
cd BBC-Text-Analysis
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook AA_Assignment.ipynb
```

Use **Python 3.13**, the version used for verification. After activation, `python` refers to the environment's interpreter. If `python3 --version` already reports 3.13, you can also use `python3` in the environment creation command. On Windows, create the environment with `py -3.13 -m venv .venv` and activate it with `.venv\Scripts\activate` instead. The saved notebook outputs can also be viewed directly on GitHub.

Install dependencies once per virtual environment, and rerun the installation command when `requirements.txt` changes. On subsequent visits, only activate the environment and open the notebook:

```bash
source .venv/bin/activate
python -m notebook AA_Assignment.ipynb
```

If you already cloned an older version, run `git pull` from your clone and install the updated requirements before opening the notebook. Select the Python kernel from `.venv`, then choose **Restart Kernel and Run All Cells** to reproduce the analysis.

### Dependencies and compatibility

[`requirements.txt`](requirements.txt) pins the tested versions of NumPy, pandas, SciPy, scikit-learn, Matplotlib, WordCloud, mlxtend, NLTK, and the Jupyter tools. The notebook imports NLTK but does not use an NLTK corpus, so no separate corpus downloads are required.

The assignment originally used Python 3.8.8. The notebook now uses current feature-name and pandas APIs, a list of stop words accepted by scikit-learn, and boolean values for Random Forest's `bootstrap` grid. Classification keeps sparse features to reduce memory, models and word clouds use fixed random seeds, and word clouds use actual keyword frequencies.

### Verify a full run

```bash
python scripts/verify_notebook.py
```

This executes every nonempty code cell in a fresh kernel using the active Python interpreter, including the full 20-fold cross-validation and grid searches. It validates the inputs, generated export, and six model test scores, then saves the refreshed notebook outputs. A detailed local report is written to `.verification/report.json`. Runtime caches, failure diagnostics, and the generated CSV are ignored by Git. Each cell has a default timeout of 1,800 seconds; use `--timeout` to change it.

## Evaluation limitations

- The vocabulary and TF-IDF weights are fitted on all articles before the train/test split and cross-validation. This introduces data leakage and can make the saved accuracy estimates optimistic. A future evaluation should fit these steps inside a training pipeline.
- Bag of Words and TF-IDF retain the assignment's different train/test split seeds. Although model randomness is now fixed, these scores still do not represent a comparison on identical test articles.
- Duplicate articles remain in the source data and can appear across evaluation splits. Very low Apriori support thresholds can also produce rules supported by only a few articles, so those associations are exploratory.


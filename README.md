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

## Analysis workflow

1. **Preprocess the text:** convert articles to lowercase and use regular expressions to remove tags, digits, and non-word characters. Load a custom stop-word list from `stopwords.txt` for feature extraction.
2. **Build text features:** create a Bag of Words matrix with up to 5,000 features, excluding terms found in more than 15% of articles, then transform it into TF-IDF weights.
3. **Extract keywords:** identify the five highest-weighted TF-IDF terms per article and export the enriched data to `bbc-text_export.csv`.
4. **Explore each category:** display keyword frequency charts and word clouds, then use Apriori and association rules to examine keywords that occur together, with support, confidence, and lift measures.
5. **Classify articles:** compare Logistic Regression, Decision Tree, and Random Forest classifiers using both feature representations, 70/30 train/test splits, 20-fold cross-validation, and grid search.

## Saved results

These accuracy scores come from the notebook's saved outputs. They have **not been reproduced in a fresh run**.

| Classifier | Features | Best cross-validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Logistic Regression | Bag of Words | 97.2% | 98.2% |
| Logistic Regression | TF-IDF | 97.4% | 98.1% |
| Decision Tree | Bag of Words | 84.5% | 85.0% |
| Decision Tree | TF-IDF | 84.4% | 85.6% |
| Random Forest | Bag of Words | 96.1% | 97.5% |
| Random Forest | TF-IDF | 96.2% | 96.3% |

Logistic Regression achieved the highest saved test accuracy, at 98.2% with Bag of Words features. See the evaluation limitations below when interpreting these results.

## Open the notebook

Clone the repository and launch Jupyter from its root directory so the relative dataset path resolves correctly:

```bash
git clone https://github.com/Goh-Wei-Chong/BBC-Text-Analysis.git
cd BBC-Text-Analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install notebook
python -m notebook AA_Assignment.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The saved notebook outputs can also be viewed directly on GitHub.

### Requirements for executing the analysis

The notebook imports NumPy, pandas, scikit-learn, Matplotlib, WordCloud, mlxtend, and NLTK:

```bash
python -m pip install numpy pandas scikit-learn matplotlib wordcloud mlxtend nltk
```

The original notebook records **Python 3.8.8**, but its package versions were not pinned. The command above lists its dependencies; executing all cells also requires the following preparation:

- Place the original `stopwords.txt` beside the notebook, with one stop word per line. This file is not currently included, so the original preprocessing cannot yet be reproduced exactly.
- Use a compatible legacy environment or update legacy APIs for current library versions, including `get_feature_names()` (see [`get_feature_names_out()`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.get_feature_names_out)) and `Series.append()` (see the [pandas removal notes](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html#removal-of-prior-version-deprecations-changes)).
- For current scikit-learn, convert the custom stop-word collection from a `frozenset` to a list before passing it to [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).
- Correct the Random Forest grid before running it on current scikit-learn: [`bootstrap` requires boolean values](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), while the original grid supplies numbers.

These are known compatibility issues; additional updates may be needed with current packages. A fresh end-to-end run has not been verified. Execute cells in order when rerunning the analysis. The grid searches use 20 folds and may take some time. The keyword export is generated during execution and is not an input file.

## Evaluation limitations

- The vocabulary and TF-IDF weights are fitted on all articles before the train/test split and cross-validation. This introduces data leakage and can make the saved accuracy estimates optimistic. A future evaluation should fit these steps inside a training pipeline.
- Bag of Words and TF-IDF use different train/test split seeds, and the Decision Tree and Random Forest models do not fix their random seeds. The scores therefore do not represent a fully controlled comparison.
- Duplicate articles remain in the source data and can appear across evaluation splits. Very low Apriori support thresholds can also produce rules supported by only a few articles, so those associations are exploratory.

## Repository files

| File | Purpose |
| --- | --- |
| [`AA_Assignment.ipynb`](AA_Assignment.ipynb) | Original analysis, visualizations, model code, and saved outputs. |
| [`bbc-text.csv`](bbc-text.csv) | Verified source dataset loaded by the notebook. |
| [`README.md`](README.md) | Project overview, dataset details, results, and execution notes. |

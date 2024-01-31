# %%
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression


# %%
trec = pd.read_csv('TREC2007.csv').drop(columns=['subject', 'email_to', 'email_from']).dropna()

spam_ham = pd.read_csv('spam_ham_dataset.csv').drop(columns=['label', 'Unnamed: 0']).dropna()
spam_ham = spam_ham.rename(columns={"text": "message", "label_num": "label"})

spam = pd.read_csv('spam.csv', encoding='latin-1').drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']).dropna()
spam = spam.rename(columns={"v1": "label", "v2": "message"})
spam['label'] = spam['label'].map({'spam': 1, 'ham': 0})

sms_spam = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message']).dropna()
sms_spam['label'] = sms_spam['label'].map({'spam': 1, 'ham': 0})

enron = pd.read_csv('enron.csv').dropna()
enron['label'] = enron['label'].map({'spam': 1, 'ham': 0})

gpt = pd.read_csv('TREC2007.csv').drop(columns=['subject', 'email_to', 'email_from']).dropna()


data = pd.concat([trec, spam_ham, spam, sms_spam, enron], ignore_index=True)
gptdata = pd.concat([data, gpt])
label = data['label']
# gptdata.head()


# %%
count_vectorizer = CountVectorizer()
TF_vectorizer = count_vectorizer.fit_transform(data['message'])

tfidf_vectorizer = TfidfVectorizer()
TFIDF_vectorizer = tfidf_vectorizer.fit_transform(data['message'])

NB_TF_class = MultinomialNB()
NB_TF_class.fit(TF_vectorizer, label)

NB_TFIDF_class = MultinomialNB()
NB_TFIDF_class.fit(TFIDF_vectorizer, label)

reg_TF_classifier = LogisticRegression(solver='sag')
reg_TF_classifier.fit(TF_vectorizer, label)

reg_TFIDF_classifier = LogisticRegression(solver='sag')
reg_TFIDF_classifier.fit(TFIDF_vectorizer, label)

rskf = RepeatedStratifiedKFold(random_state=40, n_repeats=3, n_splits=5)
NB_TF_scores = cross_val_score(NB_TF_class, TF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')
NB_TFIDF_scores = cross_val_score(NB_TFIDF_class, TFIDF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')
reg_TF_scores = cross_val_score(reg_TF_classifier, TF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')
reg_TFIDF_scores = cross_val_score(reg_TFIDF_classifier, TFIDF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')



# models with gpt data
# %%
label = gptdata['label']

gptcount_vectorizer = CountVectorizer()
gptTF_vectorizer = gptcount_vectorizer.fit_transform(gptdata['message'])

gpttfidf_vectorizer = TfidfVectorizer()
gptTFIDF_vectorizer = gpttfidf_vectorizer.fit_transform(gptdata['message'])

gptNB_TF_class = MultinomialNB()
gptNB_TF_class.fit(gptTF_vectorizer, label)

gptNB_TFIDF_class = MultinomialNB()
gptNB_TFIDF_class.fit(gptTFIDF_vectorizer, label)

gptreg_TF_classifier = LogisticRegression(solver='sag')
gptreg_TF_classifier.fit(gptTF_vectorizer, label)

gptreg_TFIDF_classifier = LogisticRegression(solver='sag')
gptreg_TFIDF_classifier.fit(gptTFIDF_vectorizer, label)

gptNB_TF_scores = cross_val_score(gptNB_TF_class, gptTF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')
gptNB_TFIDF_scores = cross_val_score(gptNB_TFIDF_class, gptTFIDF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')
gptreg_TF_scores = cross_val_score(gptreg_TF_classifier, gptTF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')
gptreg_TFIDF_scores = cross_val_score(gptreg_TFIDF_classifier, gptTFIDF_vectorizer, label, cv=rskf, scoring='balanced_accuracy')



# data visualization 
# %%
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.DataFrame({
    'TF naive Bayes': NB_TF_scores,
    'gptTF naive Bayes': gptNB_TF_scores,
    'TF-IDF naive Bayes': NB_TFIDF_scores,
    'gptTF-IDF naive Bayes': gptNB_TFIDF_scores,
    'TF logistic regression': reg_TF_scores,
    'gptTF logistic regression': gptreg_TF_scores,
    'TF-IDF logistic regression': reg_TFIDF_scores,
    'gptTF-IDF logistic regression': gptreg_TFIDF_scores
})


mean_scores = results.mean()
std_scores = results.std()
summary_table = pd.DataFrame({'Mean Accuracy': mean_scores, 'Standard Deviation': std_scores})
print("Summary of Results:")
print(summary_table)

# Visualization
plt.figure(figsize=(10, 5))
sns.boxplot(data=results)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.grid(True)
plt.show()



# %%
from scipy import stats

print(stats.ttest_ind(NB_TF_scores, gptNB_TF_scores))
print(stats.ttest_ind(NB_TFIDF_scores, gptNB_TFIDF_scores))
print(stats.ttest_ind(reg_TF_scores, gptreg_TF_scores))
print(stats.ttest_ind(reg_TFIDF_scores, gptreg_TFIDF_scores))











# collecting data from folder
# %%
import os


combined_df = pd.DataFrame()

temp_dir = 'enron/enron1'
for folder_name in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Assuming each file is a CSV with a header
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        file_content = file.read()
                    temp_df = pd.DataFrame({'message': [file_content], 'label': [folder_name]})
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

temp_dir = 'enron/enron2'
for folder_name in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Assuming each file is a CSV with a header
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        file_content = file.read()
                    temp_df = pd.DataFrame({'message': [file_content], 'label': [folder_name]})
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
temp_dir = 'enron/enron3'
for folder_name in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Assuming each file is a CSV with a header
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        file_content = file.read()
                    temp_df = pd.DataFrame({'message': [file_content], 'label': [folder_name]})
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
temp_dir = 'enron/enron4'
for folder_name in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Assuming each file is a CSV with a header
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        file_content = file.read()
                    temp_df = pd.DataFrame({'message': [file_content], 'label': [folder_name]})
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
temp_dir = 'enron/enron5'
for folder_name in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Assuming each file is a CSV with a header
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        file_content = file.read()
                    temp_df = pd.DataFrame({'message': [file_content], 'label': [folder_name]})
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
temp_dir = 'enron/enron6'
for folder_name in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Assuming each file is a CSV with a header
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        file_content = file.read()
                    temp_df = pd.DataFrame({'message': [file_content], 'label': [folder_name]})
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

combined_df.to_csv('enron.csv', index=False)





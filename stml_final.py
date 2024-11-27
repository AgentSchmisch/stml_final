#!/usr/bin/env python
# coding: utf-8

# # AICV - STML final project
# 
# Used the dataset [Phising Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
# 
# Naser Abdullah Alam. (2024). Phishing Email Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/5074342
# 
# ## Dataset details
# 
# for more depth in the data the datasets
# - CEAS
# - Nazario
# - Nigerian Fraud
# - SpamAssassin
# 
# were used 
# 
# all these datasets contain the columns
# | column   | description                                  | datatype |
# |----------|----------------------------------------------|----------|
# | sender   | sender's name and email adress                        | string   |
# | receiver | receiver's email adress                       | string   |
# | date     | date that the mail was sent                  | string   |
# | subject  | subject of the mail                          | string   |
# | body     | contents of the mail                         | string   |
# | urls     | does the message contain urls                | bool     |
# | label    | is the message a phising mail or is it legit | bool     |

# - 29.11. presentation of the topic
# - 13.12. final presentation
# 
# ---
# - contain section about ethics
# - potential biases, problems,...

# ## Data analysis & preprocessing

# In[2]:


# import the nessecary dependencies
import pandas as pd
import os
from datetime import datetime


# In[3]:


# reading the csv file from the dataset folder
path = os.path.abspath(os.getcwd())

temp_array = []


for file in os.listdir(os.path.join(path,"dataset")):
    if file.endswith(".csv"):
        temp_data = pd.read_csv(os.path.join(path,"dataset",file), dtype={
            "sender":"str",
            "receiver":"str",
            "date":"str",
            "subject":"str",
            "body":"str",
            "urls":"bool",
            "label":"bool"
        })
        temp_array.append(temp_data)
raw_data = pd.concat(temp_array, ignore_index=True)

#raw_data = pd.read_csv(os.path.join(path,"dataset/CEAS_08.csv"))

# TODO: reformat the date column to datetime
print(raw_data.info(verbose=True))


# In[4]:


# remove unnessecary columns from the dataset
final_data = raw_data[["sender", "date", "subject", "body", "urls", "label"]]

# split up the sender into sender name and sender email adress
# strip the sender_name and the sender_email from artifacts
final_data["sender_name"] = final_data["sender"].str.split("<").str[0].str.rstrip()
final_data["sender_email"] = final_data["sender"].str.split("<").str[1].str.replace(">","")
# TODO: filter out the url from the content of the mail and add a flag that represents ssl in the url (is there https in the url?)
# remove the sender column that was split up
final_data.drop(columns=["sender"], inplace=True)
# remove null values from the column

final_data = final_data.dropna()

# rearrange the data columns
final_data = final_data[["sender_name", "sender_email", "date", "subject", "body", "label","urls"]]
y = final_data["label"]
final_data.drop("label", axis=1, inplace=True)

print(final_data.info())


# ## Preprocessing
# - split the dataset into training and test sets
# - vectorize the dataset
# - 
# - get the embeddings of the tokens

# In[5]:


# use sklearn to split the dataset

from sklearn.model_selection import train_test_split

# create a train and test dataset consisting of 33% of the original dataset in the testset
X_train, X_test, y_train, y_test = train_test_split(final_data, y, test_size=0.33, random_state=42)

# Check the sizes of the splits
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ## Vectorizing
# 
# remove all the stopwords
# 
# Using sklearn.TfidfVectorizer does an equivalent operation as:
# > vectorize the data in the column subject and vectorize the data with a `CountVectorizer`
# 
# > Transform the data with a `TF-IDF Transformer` on the Counted data to weigh the occurences
#  

# In[21]:


import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

# download the stopwords package
import nltk
nltk.download("stopwords")

# Remove stop words from the raw text (before applying CountVectorizer)
# Get the stopwords list
stop_words = set(stopwords.words("english"))

# Function to remove stop words from a text
def remove_stop_words(text):
    # Split text into words, remove stop words, and rejoin the remaining words
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Apply the function to the 'subject' column and clean the text
X_train['cleaned_subject'] = X_train['subject'].apply(remove_stop_words)
X_test['cleaned_subject'] = X_test['subject'].apply(remove_stop_words)


vectorizer = TfidfVectorizer()

# Fit and transform the data
X_train_tfidf = vectorizer.fit_transform(X_train['cleaned_subject'])
X_test_tfidf = vectorizer.transform(X_test["cleaned_subject"])
# Convert the TF-IDF matrix to a DataFrame
X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Display the TF-IDF DataFrame
print(X_train_tfidf_df.info())


# In[34]:


# define a new transformer for stopword removal
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveStopWords(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.language="english"
        self.stop_words = set(stopwords.words(self.language))
        
    def fit(self, X, y=None):
        return self
        
    def remove_stop_words(self, text):
            return ' '.join([word for word in text.split() if word.lower() not in self.stop_words])

    def transform(self, X):
            # Check if X is a list; apply remove_stop_words to each element
            if isinstance(X, list):
                return [remove_stop_words(text) for text in X]
            # If X is a pandas Series, use apply
            elif isinstance(X, pd.Series):
                return X.apply(remove_stop_words)
            else:
                raise ValueError("Input type not supported. Expected list or pandas Series.")


# ## Wordcloud of Email subjects

# In[9]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


tfidf_sums = X_train_tfidf_df.sum(axis=0)
wordcloud_tfidf = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_sums)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_tfidf, interpolation='bilinear')
plt.axis('off')
plt.savefig(os.path.join(path,"wordcloud_subject.png"))
plt.show()


# ## Model Training

# In[10]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Check the shape of the training data
print(X_train_tfidf.shape, y_train.shape)  # These should match in the first dimension

# Fit the classifier
model = MultinomialNB().fit(X_train_tfidf, y_train)


# ## Create a pipeline with the model

# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

spam_clf = Pipeline([
    ("sw", RemoveStopWords()),
    ("vect", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

spam_clf.fit(X_train["subject"], y_train)


# In[17]:


from sklearn import metrics

predicted = spam_clf.predict(X_test["subject"])

cm = metrics.confusion_matrix(y_test, predicted)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=spam_clf.classes_)
disp.plot()


# # Model evaluation

# In[6]:


from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(spam_clf.predict(X_test["subject"]), y_test))


# # using a MLP Classifier

# In[7]:


from sklearn.neural_network import MLPClassifier
model = MLPClassifier()

model.fit(X_train_tfidf, y_train)


# In[11]:


from skl2onnx import to_onnx
import numpy
onx = to_onnx(model, X_train_tfidf_df[:1].astype(numpy.float32), target_opset=12)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())


# In[ ]:


y_pred = model.predict(X_test_dfidf)

print("Accuracy: ", accuracy_score(y_test, y_pred))


# # Explainability of the model

# In[61]:


from lime.lime_text import LimeTextExplainer
from random import randint
class_names = ["spam", "ham"]
print(X_test.shape)
explainer = LimeTextExplainer(class_names=class_names)
randoms = [randint(0, X_test.shape[0]) for p in range(0, 10)]


# Explaining the instance
for idx in randoms:

    # Access the subject text
    text_instance = X_test.iloc[idx]["subject"]
    print(f"Text instance: {text_instance}")
    exp = explainer.explain_instance(
        text_instance, 
        lambda x: spam_clf.predict_proba(x),  # Ensure the classifier can handle raw text
        num_features=10
    )
    
    # Print the results
    print(f"Document id: {idx}")
    probabilities = spam_clf.predict_proba([text_instance])[0]  # Correct indexing
    print(f"Probability(spam) = {probabilities[0]}")
    print(f"Probability(ham) = {probabilities[1]}")
    
    true_class_label = y_test.iloc[idx]
    print(f"True class: {class_names[true_class_label]}")
    
    exp.as_list()
    
    exp.show_in_notebook(text=True)


# # Flask server for ease of use of the algorithm

# In[45]:


# create a function that will read .msg files and return an object of the message
import extract_msg
import os

msg = extract_msg.openMsg(os.path.join(os.path.abspath(os.getcwd()),"dataset","testmails","1.msg"))
print(dir(msg))

print(msg.body)




# In[26]:


from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# HTML template for the form
form_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Form</title>
</head>
<body>
    <h1>Input Parameters</h1>
    <form action="/predict" method="post">
        <label for="param1">subject:</label>
        <input type="text" id="param1" name="param1" required><br><br>
        <label for="param2">Parameter 2:</label>
        <input type="text" id="param2" name="param2" required><br><br>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Serve the input form."""
    return render_template_string(form_template)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction based on input data."""
    # Check if the request has form data or JSON
    if request.form:
        # Get data from form
        param1 = request.form.get('param1')
        param2 = request.form.get('param2')
    elif request.json:
        # Get data from JSON body
        param1 = request.json.get('param1')
        param2 = request.json.get('param2')
    else:
        return jsonify({"error": "No data provided"}), 400

    # Mock prediction logic (replace with your own logic)
    try:
        result = int(param1) + int(param2)  # Example: summing two parameters
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Parameters should be integers."}), 400

    return jsonify({"param1": param1, "param2": param2, "prediction": result})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





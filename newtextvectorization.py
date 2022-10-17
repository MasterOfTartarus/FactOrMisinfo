import gensim
import gensim.downloader as api
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from keras import layers
import numpy as np
import transformers
import torch

# Load pre-trained model
model_, tokenizer_, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-cased')
tokenizer = tokenizer_.from_pretrained(pretrained_weights)
model = model_.from_pretrained(pretrained_weights)

# Read the CSV for article information and text
articles = pd.read_csv("Data for Misinformation - Sheet1.csv")

# Tokenize text
tokenized = articles['Article Title'].apply((lambda x: tokenizer.encode(x[-512:], add_special_tokens=True)))

# Padding so equal length
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

# Train model
input_ids = torch.tensor(np.array(padded, dtype=np.float64)).to(torch.int64)
attention_mask = torch.tensor(np.array(attention_mask, dtype=np.float64)).to(torch.int64)
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# Output features
features = np.array(last_hidden_states[0][:, 0, :])

# Reformat the text to get rid of unnecessary characters
# def fix_text(x):
#     list_corpus = x.strip(".").strip(",").strip("[").strip("]").strip(";").strip(":").split(" ")
#     return model.infer_vector(list_corpus)


# Create a list of vectors with their text fixed
# vector = []
# for art in range(article_text.shape[0]):
#     vector.append(fix_text(article_text[art]))


# Create new file with vectors (NOT NEEDED ANYMORE)
# def g(x):
#     dict_ = {True: 1, False: 0}
#     return dict_[x]
# articles['Veracity Int'] = articles['Veracity'].apply(lambda x: g(x))
#
# articles['Vector'] = articles['Article Text'].apply(lambda x: fix_text(x))
#
# articles.to_csv("Model Input.csv", index=False)


# Read new file and create variables based on the inputs and outputs of the neural network
articles = pd.read_csv("Model Input.csv")
X = features
y = articles['Veracity Int']


# Train test split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reformat all data into NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Create the sklearn classifier and train it (OBSOLETE)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)


# Define the model constants
max_features = 20000
embedding_dim = 128
sequence_length = 500



# Create the model
misinfo = keras.models.Sequential()
misinfo.add(keras.layers.Flatten(input_shape=[768,])) # X_train has 40 features
misinfo.add(keras.layers.Dense(300, activation='relu'))
misinfo.add(keras.layers.Dense(100, activation='relu'))
misinfo.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
misinfo.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
epochs = 30

misinfo.fit(X_train, y_train, epochs=epochs)


# Test the model's accuracy
misinfo.evaluate(X_test, y_test)

# Input your own statement

_tokenizer_ = tokenizer.encode("Donald Trump is BEST president ever and Barack Obama is a muslim terrorist",
                               add_special_tokens=True)
_padded = np.array(_tokenizer_ + [0] * (max_len - len(_tokenizer_)))
_attention_mask = np.where(_padded != 0, 1, 0)

# Train model
_input_ids = torch.tensor(np.array(_padded, dtype=np.float64)).to(torch.int64).unsqueeze(0)
_attention_mask = torch.tensor(np.array(_attention_mask, dtype=np.float64)).to(torch.int64).unsqueeze(0)
with torch.no_grad():
    last_hidden_states_ = model(_input_ids, attention_mask=_attention_mask)
statement = np.array(last_hidden_states_[0][:, 0, :])

print(misinfo.predict(statement))

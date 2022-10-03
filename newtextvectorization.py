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

# Load the vectorization model
dataset = api.load("text8")
data = [d for d in dataset]


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


data_for_training = list(tagged_document(data))

model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
model.build_vocab(data_for_training)
model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)

# Read the CSV for article information and text
articles = pd.read_csv("Data for Misinformation - Sheet1.csv")
article_text = articles["Article Text"]


# Reformat the text to get rid of unnecessary characters
def fix_text(x):
    list_corpus = x.strip(".").strip(",").strip("[").strip("]").strip(";").strip(":").split(" ")
    return model.infer_vector(list_corpus)


# Create a list of vectors with their text fixed
vector = []
for art in range(article_text.shape[0]):
    vector.append(fix_text(article_text[art]))


# Create new file with vectors
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
X = vector
y = articles['Veracity Int']


# Train test split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reformat all data into NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Put the data into tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# Create the sklearn classifier and train it
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)


# Define the model constants
max_features = 20000
embedding_dim = 128
sequence_length = 500


# Create a Keras neural network
# inputs = keras.Input(shape=(None,), dtype="int64")
#
# z = layers.Embedding(max_features, embedding_dim)(inputs)
# z = layers.Dropout(0.5)(z)
#
# z = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(z)
# z = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(z)
# z = layers.GlobalMaxPooling1D()(z)
#
# # Vanilla hidden layer
# z = layers.Dense(128, activation="relu")(z)
# z = layers.Dropout(0.5)(z)
#
# predictions = layers.Dense(1, activation="sigmoid", name="predictions")(z)


# Create the model
# model = keras.Model(inputs, predictions)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(None,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# Train the model
epochs = 3

# print(f"{X_train=}")
# print(f"{y_train=}")
# print(f"{X_test=}")
# print(f"{y_test=}")
#
# print()
# print()
#
# print(len(X_train))
# print(len(y_train))
# print(len(X_test))
# print(len(y_test))
#
# print(X_train.shape)
# print(y_train.shape)

model.fit(train_dataset, epochs=epochs)


# Test the model's accuracy
model.evaluate(test_dataset)

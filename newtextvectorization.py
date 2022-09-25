import gensim
import gensim.downloader as api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

dataset = api.load("text8")
data = [d for d in dataset]


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


data_for_training = list(tagged_document(data))

model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
model.build_vocab(data_for_training)
model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)

articles = pd.read_csv("Data for Misinformation - Sheet1.csv")
article_text = articles["Article Text"]

def f(x):
    list_corpus = x.strip(".").strip(",").strip("[").strip("]").strip(";").strip(":").lower().split(" ")
    return model.infer_vector(list_corpus)

vector = []
for art in range(article_text.shape[0]):
    vector.append(f(article_text[art]))

# def g(x):
#     dict_ = {True: 1, False: 0}
#     return dict_[x]
# articles['Veracity Int'] = articles['Veracity'].apply(lambda x: g(x))
#
# articles['Vector'] = articles['Article Text'].apply(lambda x: f(x))
#
# articles.to_csv("Model Input.csv", index=False)

articles = pd.read_csv("Model Input.csv")
X = vector
y = articles['Veracity Int']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)

training_score = clf.predict(X_train)
test_score = clf.predict(X_test)

test_accuracy = accuracy_score(y_test, test_score)
print(test_accuracy)
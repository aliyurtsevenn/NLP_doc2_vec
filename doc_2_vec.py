import  pandas as pd
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


path_="../spam.csv"

data_=pd.read_csv(path_,encoding="latin-1")
data_=data_.drop(labels=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
data_.columns= ["labels","text"]

# Let's clean the data!
print(data_)

data_["text_clean"]= data_["text"].apply(lambda x: gensim.utils.simple_preprocess(x))

X_train, X_test, Y_train, Y_test= train_test_split(data_["text_clean"], data_["labels"],
                                                   test_size=0.2)

# Here, we need to create a document tag before training our data!

tagged_doc = [gensim.models.doc2vec.TaggedDocument(v,[i]) for i, v in enumerate(X_train)]

# Let me look at what the tagged document looks like!
print(tagged_doc[0])

# Let me train a basic doc2vec model
d2v_model = gensim.models.Doc2Vec(
    tagged_doc,
    vector_size=100,
    window=5,
    min_count=2
)

# Let me check list of strings to find its vector representation!

names_vector = d2v_model.infer_vector(["i","am","learning","nlp"])

print(names_vector)


# How to prepare these vectors to be used in a machine learning model?
vectors= [[d2v_model.infer_vector(words)] for words in X_test]
print(vectors[0])


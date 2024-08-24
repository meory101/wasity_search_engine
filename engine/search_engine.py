import requests
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


response =  requests.get('http://127.0.0.1:8000/api/getProducts')
data =response.content.decode('utf-8')
data_dict = json.loads(data)
if response.status_code == 200:
    df = pd.DataFrame(data_dict)
    print(df)

class IndexModel:
    def __init__(self, documents_df):

        self.tfidf_vectorizer = TfidfVectorizer()
        self._index = self.tfidf_vectorizer.fit_transform(df["desc"])

    def getindex(self):
        return self._index

    def vectorize(self, sentence):
        if isinstance(sentence,str):
            qry=pd.DataFrame([{"desc":sentence}])
        else:
            qry=sentence
        return self.tfidf_vectorizer.transform(qry['desc'])



class Retriever:

    def retrieve(self, query_vec, index_model):
        cosine_similarities = cosine_similarity(query_vec, index_model.getindex())
        results = pd.DataFrame(
            [{'docno':df['id'][i], 'score':cosine_similarities[0][i], 'content':None}
            for i in range(len(cosine_similarities[0]))]
        ).sort_values(by=['score'], ascending=False)
        return results[results["score"]>0]
    




rt = Retriever()
vsm = IndexModel(df)
v=vsm.getindex()
qrv=vsm.vectorize('skskks')
res = rt.retrieve(qrv,vsm)
print(res) 
from nltk.corpus import abc
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
dataset = abc.raw('science.txt')

#queries to summarize
queries = ["What is DNA and RNA?",
         "What is the largest animal?",
         "How to prevent a heart attack?"]

#function for query/question processing (module) - tokenize
def process_query(query):
    tokens = word_tokenize(query)
    return tokens

#function for sentence selection (module), taking dataset, query and number of sentences
#using tf-idf and cosine to calculate
def select_sentences(dataset, tokens, num_sentences=2):
    #first tokenize dataset
    sentences = sent_tokenize(dataset)

    #TF-IDF vectorizer from scikit-learn
    tf_idf = TfidfVectorizer()
    tfidf_st = tf_idf.fit_transform(sentences + [tokens])

    #cosine calculation for sentences and query to determine how similar
    query_c = tfidf_st[-1]
    calcs = cosine_similarity(tfidf_st[:-1], query_c)

    #save top sentences that match
    matches = calcs.flatten().argsort()[-num_sentences:][::-1]
    top_sent = [sentences[i] for i in matches]

    return top_sent

#summarizer module to generate final summary
def summary_module(selected):
    final_summary = ' '.join(selected)
    return final_summary

#for loop to go through queries and create summary through modules
for query in queries:
    selected = select_sentences(dataset, query)
    summary = summary_module(selected)

#print query followed by top sentences and final summary
    print(f"Query: {query}")
    print("Top Selected Sentences:")
    for sentence in selected:
        print(sentence)
    print("Summary:" + summary)
    print("--------------------------")
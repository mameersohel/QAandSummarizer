import numpy as np
from nltk.corpus import abc
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
#create document and question objects
document = abc.raw("science.txt")
question = ("What is the largest animal?")

#function for question processing (module) to tokenize
def question_processing(question):
    tokens = word_tokenize(question)
    return tokens

#create question tokens from function
question_tokens = question_processing(question)

#ranking module to rank top answer from matches (passage retrieval)
def ranking(question_tokens, document):
    #segmenting and creating paragraphs
    paragraphs = document.split('\n')

    #TF-IDF vectorizer from scikit-learn
    tf_idf = TfidfVectorizer()
    tfidf_paragraphs = tf_idf.fit_transform(paragraphs)

    #calculate TF-IDF scores from questions tokens
    question_scores = tf_idf.transform([" ".join(question_tokens)])

    #cosine similarities to find similarities between tf-idf of question and dataset
    cos_similarities = np.dot(tfidf_paragraphs, question_scores.T).toarray().ravel()

    #ranking paragraphs in list based on cosine similarities scores (descending order)
    ranked_answers = [(paragraph, score) for paragraph, score in zip(paragraphs, cos_similarities)]
    ranked_answers.sort(key=lambda x: x[1], reverse=True)

    return ranked_answers


#call function to select answer and place in object (answer selection module)
answer = ranking(question_tokens, document)
if answer:
    answer = answer[0]
else:
    answer = "Not in dataset"

#print question and answer
print(question)
print("Answer:")
print(answer)
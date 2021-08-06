# -------------------Part 1 of Yelp Project: Create a Class to Pre-Process reviews and ratings------------------

# First, import essential packages
import requests
import bs4
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
sns.set()




wnl = WordNetLemmatizer()
connectors = list(set(stopwords.words("english")))
allowed_modifiers = ['J', 'R', 'C']

# Create a class blueprint to track the pages
class YelpProcess:
    
    def __init__(self, link):
        self.link = link
        self.reviews = []
        self.ratings = []
        self.labels = []
   
        
    def process_reviews_ratings(self):
        page = requests.get(self.link)
        soup = BeautifulSoup(page.text, 'html.parser')
        
        # Extract the div-tags that contain the (MAIN) review content
        div_tags = soup.findAll('div', {'class': 'review-content' })
        
        # Extract all the paragraph <p lang="en"> tags (each one contains a review) from div-tags
        reviews_tag = [tag.p for tag in div_tags]
        raw_contents = [r.contents for r in reviews_tag]
        filtered = [[line for line in C if type(line) == bs4.element.NavigableString] for C in raw_contents]
        reviews_raw = [' '.join(f) for f in filtered]
        
        # Extract all the corresponding ratings for each review
        ratings = [float(tag.div.div.div.img['alt'].split()[0]) for tag in div_tags]
        self.ratings = ratings
        
        # Now, the review has been loaded into Python from Yelp! Final Processing of reviews
        reviews_lower = [r.lower() for r in reviews_raw]
        reviews_tok = [word_tokenize(r) for r in reviews_lower]
        reviews_words = [[w for w in token if w.isalpha() == True] for token in reviews_tok]
        # Alternative 1: below (take out stopwords)
        #reviews_filtered = [[w for w in parsed if w not in connectors] for parsed in reviews_words]
        #reviews = [' '.join(edited) for edited in reviews_filtered]
        # Alternative 2: below (do NOT take out stopwords)
        #reviews = [' '.join(edited) for edited in reviews_words]
        # Alternative 3: Part-Of-Speech tagging to isolate adjectives, adverbs, conjs.
        reviews_pos = [pos_tag(W) for W in reviews_words]
        reviews_keywords = [[pair[0] for pair in P if pair[1][0] in allowed_modifiers] for P in reviews_pos]
        reviews = [' '.join(k_list) for k_list in reviews_keywords]
        self.reviews = reviews
        
    
    
    def make_labels(self):
        labels = []
        for rate in self.ratings:
            if rate>= 4.0 and rate <= 5.0:
                labels.append(2)
            elif rate == 3.0:
                labels.append(1)
            elif rate < 3.0:
                labels.append(0)
        self.labels = labels
    
# We need the labels to be converted into numbers
# let 'Good' = 2
# let 'Average' = 1
# let 'Bad' = 0




#----------------------- Part 2 of Yelp Project: Analysis of Reviews and matching ratings ------------------------

# ---------------- Section 1: Make the DataFrame --------------------

# Now that we have created the initial website reviews/ratings Class structure,
# we need to input the reviews and ratings into a Pandas Dataframe

# ------------ The training and testing data is below -----------

# These are the 5 restaurants that a total of 1,720 reviews have been drawn from:
# https://www.yelp.com/biz/aki-japanese-restaurant-berkeley (81 reviews)
# https://www.yelp.com/biz/kaze-ramen-berkeley-3 (375 reviews)
# https://www.yelp.com/biz/muraccis-berkeley-berkeley (257 reviews)
# https://www.yelp.com/biz/tako-sushi-berkeley (541 revs)
# https://www.yelp.com/biz/berkeley-thai-house-berkeley (453 reviews)

p1 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley"
p2 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley?start=20"
p3 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley?start=40"
p4 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley?start=60"
p5 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley"
p6 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=20"
p7 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=40"
p8 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=60"
p9 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=80"
p10 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=100"
p11 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=120"
p12 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=140"
p13 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=160"
p14 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=180"
p15 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=200"
p16 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=220"
p17 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=240"
p18 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3"
p19 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=20"
p20 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=40"
p21 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=60"
p22 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=80"
p23 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=100"
p24 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=120"
p25 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=140"
p26 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=160"
p27 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=180"
p28 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=200"
p29 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=220"
p30 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=240"
p31 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=260"
p32 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=280"
p33 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=300"
p34 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=320"
p35 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=340"
p36 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=360"
p37 = "https://www.yelp.com/biz/tako-sushi-berkeley"
p38 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=20"
p39 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=40"
p40 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=60"
p41 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=80"
p42 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=100"
p43 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=120"
p44 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=140"
p45 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=160"
p46 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=180"
p47 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=200"
p48 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=220"
p49 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=240"
p50 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=260"
p51 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=280"
p52 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=300"
p53 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=320"
p54 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=340"
p55 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=360"
p56 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=380"
p57 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=400"
p58 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=420"
p59 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=440"
p60 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=460"
p61 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=480"
p62 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=500"
p63 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=520"
p64 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley"
p65 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=20"
p66 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=40"
p67 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=60"
p68 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=80"
p69 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=100"
p70 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=120"
p71 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=140"
p72 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=160"
p73 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=180"
p74 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=200"
p75 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=220"
p76 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=240"
p77 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=260"
p78 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=280"
p79 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=300"
p80 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=320"
p81 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=340"
p82 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=360"
p83 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=380"
p84 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=400"
p85 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=420"
p86 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=440"

# --------------- End of data haul from Yelp ----------------



def review_setup(link):
    R = YelpProcess(link)
    R.process_reviews_ratings()
    R.make_labels()
    return R
    
def review_objects(links):
    result = []
    for i in range(0, len(links)):
        result.append(review_setup(links[i]))
    return result
        
def review_frame(Obj):
    df = pd.DataFrame({'reviews': Obj.reviews, 'opinions': Obj.labels})
    return df


def make_frames(obj_list):
    frames=[]
    for obj in obj_list:
        frames.append(review_frame(obj))
    return frames



pages = [p1,p2,p3,p4,p5,p6,p7,p8,p9,
         p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,
         p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,
         p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,
         p40,p41,p42,p43,p44,p45,p46,p47,p48,p49,
         p50,p51,p52,p53,p54,p55,p56,p57,p58,p59,
         p60,p61,p62,p63,p64,p65,p66,p67,p68,p69,
         p70,p71,p72,p73,p74,p75,p76,p77,p78,p79,
         p80,p81,p82,p83,p84,p85,p86]

link_objs = review_objects(pages)
all_reviews = pd.concat(make_frames(link_objs)).reset_index(drop=True)      







# -------------- This section is CountVectorizer: DTM Matrix (word-frequency) Analysis --------------
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(max_features=5000, binary=True)
sparse_dtm = countvec.fit_transform(all_reviews['reviews'])
features_dtm = sparse_dtm.toarray()


# ---------------- Now we will use the Logistic (Multi-Class) Classifier and create our training/test data -------

response = all_reviews['opinions'].values

# Part 1: First, we will do our Logit Classifier on the DTM features matrix (Count Vectorizer) ---
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

X_dtm, y_dtm = features_dtm, response

# First we need to split our data into training and testing sets
X_train_dtm, X_test_dtm, y_train_dtm, y_test_dtm = train_test_split(X_dtm, y_dtm, train_size=0.9, random_state=0)

# Classifier for DTM data
logit_dtm_OVR = OneVsRestClassifier(LogisticRegressionCV())
logit_dtm_OVO = OneVsOneClassifier(LogisticRegressionCV())
OVR = logit_dtm_OVR.fit(X_train_dtm, y_train_dtm)
OVO = logit_dtm_OVO.fit(X_train_dtm, y_train_dtm)

OVR_score = logit_dtm_OVR.score(X_test_dtm, y_test_dtm)
OVO_score = logit_dtm_OVO.score(X_test_dtm, y_test_dtm)

print("One vs rest accuracy (DTM): {}%".format(round(100*OVR.score(X_test_dtm,y_test_dtm), 3)))
print("One vs one accuracy (DTM): {}%".format(round(100*OVO.score(X_test_dtm,y_test_dtm), 3)))



# This part is for graphing the accuracy of the OVR-classifier (DTM)
labels_dtm_ovr = OVR.classes_
y_pred_dtm_ovr = OVR.predict(X_test_dtm)
cm_dtm_ovr = confusion_matrix(y_test_dtm, y_pred_dtm_ovr)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_ovr, annot=True, xticklabels=labels_dtm_ovr, yticklabels=labels_dtm_ovr)
plt.xlabel('Prediction (OVR)')
plt.ylabel('Truth')

# This part is for graphing the accuracy of the OVO-classifier (DTM)
labels_dtm_ovo = OVO.classes_
y_pred_dtm_ovo = OVO.predict(X_test_dtm)
cm_dtm_ovo = confusion_matrix(y_test_dtm, y_pred_dtm_ovo)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_ovo, annot=True, xticklabels=labels_dtm_ovo, yticklabels=labels_dtm_ovo)
plt.xlabel('Prediction (OVO)')
plt.ylabel('Truth')


# ------------- Calculating the Precision and Recall scores OVR classifier (DTM-features) -----------
from sklearn.metrics import precision_recall_fscore_support as scores

precision, recall, fscore, support = scores(y_test_dtm, y_pred_dtm_ovr)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# ------------- Calculating the Precision and Recall scores OVO classifier (DTM-features) -----------
from sklearn.metrics import precision_recall_fscore_support as scores

precision, recall, fscore, support = scores(y_test_dtm, y_pred_dtm_ovo)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))





# ---------------- Section 2: TF-IDF Matrix Vectorizer to analyze Word importance -----------------

# First, create the TF-IDF Vectorizer Object
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
sparse_tfidf = tfidfvec.fit_transform(all_reviews['reviews'])

#Next, create the TF-IDF feature matrix
#tfidf = pd.DataFrame(sparse_tfidf.toarray(), columns=tfidfvec.get_feature_names(), index=all_reviews.index)
features_tfidf = sparse_tfidf.toarray()



# Part 2: Now, we will do our Logit Classifier on the TF-IDF features matrix ---------
# We are hoping to use word features from the TF-IDF matrix to help
# First we need to split our data into training and testing sets
X_tfidf, y_tfidf = features_tfidf, response

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, train_size=0.9, random_state=1)


# Logit Classifier for TF-IDF features/data
logit_tfidf_OVR = OneVsRestClassifier(LogisticRegressionCV())
logit_tfidf_OVO = OneVsOneClassifier(LogisticRegressionCV())
OVR = logit_tfidf_OVR.fit(X_train_tfidf, y_train_tfidf)
OVO = logit_tfidf_OVO.fit(X_train_tfidf, y_train_tfidf)
# Scoring
OVR_score = logit_tfidf_OVR.score(X_test_tfidf, y_test_tfidf)
OVO_score = logit_tfidf_OVO.score(X_test_tfidf, y_test_tfidf)
# Print accuracies
print("One vs rest accuracy (TF-IDF): {}%".format(round(100*OVR.score(X_test_tfidf, y_test_tfidf), 3)))
print("One vs one accuracy (TF-IDF): {}%".format(round(100*OVO.score(X_test_tfidf, y_test_tfidf), 3)))


# This last part is for graphing the accuracy of the classifier (TF-IDF) using OVR classification
labels_tfidf_ovr = OVR.classes_
y_pred_tfidf_ovr = OVR.predict(X_test_tfidf)
cm_tfidf_ovr = confusion_matrix(y_test_tfidf, y_pred_tfidf_ovr)

plt.figure(figsize=(10,7))
sns.heatmap(cm_tfidf_ovr, annot=True, xticklabels=labels_tfidf_ovr, yticklabels=labels_tfidf_ovr)
plt.xlabel('Prediction (OVR)')
plt.ylabel('Truth')

# This last part is for graphing the accuracy of the classifier (TF-IDF) using OVO classification

labels_tfidf_ovo = OVO.classes_
y_pred_tfidf_ovo = OVO.predict(X_test_tfidf)
cm_tfidf_ovo = confusion_matrix(y_test_tfidf, y_pred_tfidf_ovo)

plt.figure(figsize=(10,7))
sns.heatmap(cm_tfidf_ovo, annot=True, xticklabels=labels_tfidf_ovo, yticklabels=labels_tfidf_ovo)
plt.xlabel('Prediction (OVO)')
plt.ylabel('Truth')


# ------------- Calculating the Precision and Recall scores OVO classifier (TF-IDF-features) -----------
from sklearn.metrics import precision_recall_fscore_support as scores

precision, recall, fscore, support = scores(y_test_tfidf, y_pred_tfidf_ovo)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# ------------- Calculating the Precision and Recall scores OVR classifier (TF-IDF-features) -----------
from sklearn.metrics import precision_recall_fscore_support as scores

precision, recall, fscore, support = scores(y_test_tfidf, y_pred_tfidf_ovo)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))




# -------------- Now we will try doing classification based on SVM Classifier ------------



# SVM stands for Support Vector Machine algorithm

# -------------- This section is CountVectorizer: DTM Matrix (word-frequency) Analysis --------------
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(max_features=5000, binary=True)
sparse_dtm = countvec.fit_transform(all_reviews['reviews'])
features_dtm = sparse_dtm.toarray()

response = all_reviews['opinions'].values


X_dtm, y_dtm = features_dtm, response

# First we will do the SVM classification with the DTM-feature matrix

#First, split the X_dtm and y_dtm data
X_train_dtm_svm, X_test_dtm_svm, y_train_dtm_svm, y_test_dtm_svm = train_test_split(X_dtm, y_dtm, train_size=0.9, random_state=2)


# Next, import the SVM classifier and set up the model
from sklearn import svm

clf_svm_dtm = svm.SVC(gamma=0.001, C=200)
clf_svm_dtm.fit(X_train_dtm_svm, y_train_dtm_svm)


print("SVM (DTM features) Accuracy: {}%".format(round(100*clf_svm_dtm.score(X_test_dtm_svm, y_test_dtm_svm), 3)))

# This part is for graphing the accuracy of the SVM-classifier (DTM features)
labels_dtm_svm = clf_svm_dtm.classes_
y_pred_dtm_svm = clf_svm_dtm.predict(X_test_dtm_svm)
cm_dtm_svm = confusion_matrix(y_test_dtm_svm, y_pred_dtm_svm)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_svm, annot=True, xticklabels=labels_dtm_svm, yticklabels=labels_dtm_svm)
plt.xlabel('Prediction (SVM)')
plt.ylabel('Truth')


# ------------- Calculating the Precision and Recall scores (DTM-features) -----------
from sklearn.metrics import precision_recall_fscore_support as scores

precision, recall, fscore, support = scores(y_test_dtm_svm, y_pred_dtm_svm)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))





# Now we will do the SVM classification with the TF-IDF-feature matrix

# First, create the TF-IDF Vectorizer Object
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
sparse_tfidf = tfidfvec.fit_transform(all_reviews['reviews'])

#Next, create the TF-IDF feature matrix
#tfidf = pd.DataFrame(sparse_tfidf.toarray(), columns=tfidfvec.get_feature_names(), index=all_reviews.index)
features_tfidf = sparse_tfidf.toarray()



# Part 2: Now, we will do our SVM Classifier on the TF-IDF features matrix ---------
# We are hoping to use word features from the TF-IDF matrix to help
# First we need to split our data into training and testing sets
response = all_reviews['opinions'].values

X_tfidf, y_tfidf = features_tfidf, response




#First, split the X_tfidf and y_tfidf data
X_train_tfidf_svm, X_test_tfidf_svm, y_train_tfidf_svm, y_test_tfidf_svm = train_test_split(X_tfidf, y_tfidf, train_size=0.9, random_state=3)


# Next, import the SVM classifier and set up the model
from sklearn import svm

clf_svm_tfidf = svm.SVC(gamma=0.001, C=200)
clf_svm_tfidf.fit(X_train_tfidf_svm, y_train_tfidf_svm)


print("SVM (TF-IDF features) Accuracy: {}%".format(round(100*clf_svm_tfidf.score(X_test_tfidf_svm, y_test_tfidf_svm),3)))

# This part is for graphing the accuracy of the SVM-classifier (TF-IDF features)
labels_tfidf_svm = clf_svm_tfidf.classes_
y_pred_tfidf_svm = clf_svm_tfidf.predict(X_test_tfidf_svm)
cm_svm_tfidf = confusion_matrix(y_test_tfidf_svm, y_pred_tfidf_svm)

plt.figure(figsize=(10,7))
sns.heatmap(cm_svm_tfidf, annot=True, xticklabels=labels_tfidf_svm, yticklabels=labels_tfidf_svm)
plt.xlabel('Prediction (SVM)')
plt.ylabel('Truth')



# ------------- Calculating the Precision and Recall scores -----------
from sklearn.metrics import precision_recall_fscore_support as scores

precision, recall, fscore, support = scores(y_test_tfidf_svm, y_pred_tfidf_svm)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


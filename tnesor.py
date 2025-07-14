import pickle
import pandas as pd
from sklearn.model_selection import train_test_split    #split dataset into training set
from sklearn.feature_extraction.text import TfidfVectorizer  #converts text to machine understandable numbers
from sklearn.naive_bayes import MultinomialNB    #online learning model that detects spam
from sklearn.metrics import accuracy_score      #Accuracy = number of correct predictions / Total number of predictions

#Load Datasets
true_df=pd.read_csv(r"C:\Users\ABISHEK RAJ\Desktop\MachineLearning\Learn\True.csv")
fake_df=pd.read_csv(r"C:\Users\ABISHEK RAJ\Desktop\MachineLearning\Learn\Fake.csv")

#add labels
true_df["label"]=1 #Real
fake_df["label"]=0 #fake

#combine and shuffle
df = pd.concat([true_df,fake_df])
print(df['label'].value_counts())
df=df.sample(frac=1).reset_index(drop=True)

#features and labels
X=df["text"]
Y=df["label"]

#split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#tf-idf             term frequency=number of times t appears in document / total number of terms in document
#idf (inverse document frequency)=log(total number of documents / number of documents with term t)
vectorizer= TfidfVectorizer(stop_words="english",max_df=0.7)     #creates a vector tf-idf vectorizes into two tweaks
X_train_tfidf=vectorizer.fit_transform(X_train)     #transforms the text data to tf-idf vectors
X_test_tfidf=vectorizer.transform(X_test)          #transforms the text data to tf-idf vectors

#model
model = MultinomialNB()
model.fit(X_train_tfidf,Y_train)

#evaluate
Y_pred=model.predict(X_test_tfidf)
accuracy=accuracy_score(Y_test,Y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

while True:
    print("\nEnter a news article (type 'exit' to quit):")
    news = input("> ")

    if news.lower() == "exit":
        print("Exiting... 🫡")
        break

    news_tfidf = vectorizer.transform([news])
    pred = model.predict(news_tfidf)[0]

    if pred == 1:
        print("🟢 Prediction: REAL NEWS")
    else:
        print("🔴 Prediction: FAKE NEWS")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

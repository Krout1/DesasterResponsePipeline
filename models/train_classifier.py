import sys
import pandas as pd
import numpy as np
import string
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, make_scorer
import pickle

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        try:
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            return False
        except:
            return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y, Y.columns.values


def tokenize(text):
    text = text.lower().strip()
    translate_table = dict((ord(char), None) for char in string.punctuation)   
    text= text.translate(translate_table)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens=[]
    for toke in tokens:
        cleaned_toke = lemmatizer.lemmatize(toke)
        cleaned_tokens.append(cleaned_toke)
    return cleaned_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('start_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    Y_test = np.array(Y_test)
    metrics=[]
    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test[:, i], y_pred[:, i])
        precision = precision_score(Y_test[:, i], y_pred[:, i])
        recall = recall_score(Y_test[:, i], y_pred[:, i])
        f1 = f1_score(Y_test[:, i], y_pred[:, i])
        print("{0} {1} {2} {3} ".format(accuracy, precision,recall,f1))
        
   
    pass
      


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Define default arguments
default_args = {
 'owner': 'your_name',
 'start_date': datetime (2023, 9, 29),
 'retries': 1,
}

# Instantiate your DAG
dag = DAG ('model_train_1', default_args=default_args, schedule_interval=None)

 

def cleaning(df):
    df['title'] = df['title'].str.lower()
    df['text'] = df['text'].str.lower()
    df['title'] = df['title'].replace(r"\W", " ", regex=True)
    df['text'] = df['text'].replace(r"\W", " ", regex=True)
    return df

def aggregate(df):
    df["msg"] = df["title"] + " " + df["text"]
    return df

def remove_extra_col(df):
    cols = df.columns
    to_remove = [x for x in cols if x not in ["msg", "type"]]
    if len(to_remove) > 0:
        df = df.drop(to_remove, axis=1)
    return df


def data_preprocessing():
    df = pd.read_csv("./data/email_spam.csv")
    df = cleaning(df)
    df = aggregate(df)
    df = remove_extra_col(df)
    df.to_csv("./artifacts/cleaned_data.csv", index=False)


def create_artifacts(df):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['msg'])
    joblib.dump(vectorizer, "./artifacts/vectorizer.sav")

    encoder = LabelEncoder()
    encoder.fit(df['type'])
    joblib.dump(encoder, "./artifacts/encoder.sav")


def prepare_train_data(df):
    vectorizer = joblib.load("./artifacts/vectorizer.sav")
    encoder = joblib.load("./artifacts/encoder.sav")
    X = vectorizer.transform(df['msg'])
    y = encoder.transform(df['type'])
    return X, y

def train(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(train_X, train_y)
    joblib.dump(model, "./artifacts/model.sav")

    y_pred = model.predict(test_X)
    acc_val = accuracy_score(test_y, y_pred)
    return acc_val

def train_model():
   df = pd.read_csv("./artifacts/cleaned_data.csv")
   create_artifacts(df)
   X, y = prepare_train_data(df)
   accuracy = train(X, y)
   print(f"Accuracy: {accuracy}")


data_preprocessing_task = PythonOperator(
 task_id='data_preprocessing',
 python_callable=data_preprocessing,
 dag=dag,
)

train_model_task = PythonOperator(
 task_id='train_model',
 python_callable=train_model,
 dag=dag,
)


create_image_task = BashOperator(
   task_id='create_image',
   bash_command='docker build . -t spam-classifier && docker run -p 8000:8000 -d spam-classifier',
   dag=dag,
   cwd="./"
)


# Set task dependencies
data_preprocessing_task >> train_model_task >> create_image_task
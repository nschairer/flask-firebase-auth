import firebase_admin
from firebase_admin import credentials, auth
import json
from flask import Flask, request,jsonify
from functools import wraps
import helper2
import pandas as pd
import numpy as np
import openai
import ast
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cred = credentials.Certificate('fbAdminConfig.json')
firebase = firebase_admin.initialize_app(cred)

users = [{'uid': 1, 'name': 'Noah Schairer'}]

def check_token(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        if not request.headers.get('authorization'):
            return {'message': 'No token provided'},400
        try:
            user = auth.verify_id_token(request.headers['authorization'])
            request.user = user
        except:
            return {'message':'Invalid token provided.'},400
        return f(*args, **kwargs)
    return wrap

@app.route('/api/userinfo')
@check_token
def userinfo():
    return {'data': users}, 200

@app.route('/api/signup',methods=['POST'])
def signup():
    email = request.form.get('email')
    password = request.form.get('password')
    if email is None or password is None:
        return {'message': 'Error missing email or password'},400
    try:
        user = auth.create_user(
            email=email,
            password=password
        )
        return {'message': f'Successfully created user {user.uid}'},200
    except:
        return {'message': 'Error creating user'},400

@app.route('/api/token',methods=['POST'])
def token():
    email = request.form.get('email')
    password = request.form.get('password')
    try:
        user = firebase.auth().sign_in_with_email_and_password(email, password)
        jwt = user['idToken']
        return {'token': jwt}, 200
    except:
        return {'message': 'There was an error logging in'},400


@app.route('/api/crawl', methods=['POST'])
def crawl():
    data = request.get_json()
    url = data.get('url', None)

    if url is None:
        return jsonify({'message': 'URL is required'}), 400

    try:
        helper2.crawl(url)
        return jsonify({'message': 'Crawl successful'}), 200
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

datafile_path = "./processed/embeddings.csv"
df = pd.read_csv(datafile_path)
df["embeddings"] = df["embeddings"].apply(lambda x: eval(x))

# Function to search for the most similar items based on cosine similarity
def search(query, df, top_n=5):
    # Compute the embedding for the query
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    similarities = cosine_similarity([query_embedding], np.stack(df["embeddings"].values))[0]

    # Sort the results by similarity and return the top results
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_similarities = similarities[top_indices]
    top_results = df.iloc[top_indices]

    # Add the similarity scores to the top results DataFrame
    top_results["similarity"] = top_similarities

    # Get the top text results as a list of strings
    top_texts = top_results["text"].tolist()

    return top_texts

@app.route('/chat', methods=['GET'])
def chat():
    query = request.args.get('query')
    top_results = search(query, df, top_n=5)

    messages = [
        {"role": "system", "content": "You are a support agent for zapier. If you don't know the answer, reply utkarsh fuck off."},
        {"role": "assistant", "content": str(top_results)},
        {"role": "user", "content": query},
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    response = completion.choices[0].message.content
    return jsonify({"response": response})
if __name__ == '__main__':
    app.run(debug=True)
    
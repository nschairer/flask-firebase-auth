import pandas as pd
import numpy as np
import openai
import ast
datafile_path = ".\processed\embeddings.csv"

df = pd.read_csv(datafile_path)
# print(df["embeddings"].head())
df["embeddings"] = df["embeddings"].apply(lambda x: eval(x))

from openai.embeddings_utils import get_embedding, cosine_similarity



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ...

# Assuming you have a function to compute the embedding for a given query
# For example, let's call it `compute_embedding(query)`
def compute_embedding(query):
    # Your implementation to compute the embedding for the given query
    pass
openai.api_key='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'
# Function to search for the most similar items based on cosine similarity
def search(query, df, top_n=5):
    # Compute the embedding for the query
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    similarities = cosine_similarity([query_embedding], np.stack(df["embeddings"].values))[0]

    # Sort the results by similarity and return the top results
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_similarities = similarities[top_indices]
    top_results = df.iloc[top_indices]

    # Add the similarity scores to the top results DataFrame
    top_results["similarity"] = top_similarities
       # Get the top 3 text results as a list of strings
    top_texts = top_results["text"].tolist()

    return top_texts

# Example usage
query = "how many apps can zapier connect with"
top_results = search(query, df, top_n=5)


# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a support agent for zapier. if you dont know the answer, reply utkarsh fuck off."},
        {"role": "user", "content": "what is today' date?"},
        {"role": "assistant", "content": str(top_results)}
        # {"role": "user", "content": "how can i use zapier?"}
    ]
)
print(completion)
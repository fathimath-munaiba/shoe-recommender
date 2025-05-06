#!/usr/bin/env python
# coding: utf-8

# # SoleSelect (shoe recomender system)

# ## 1. Importing libraries and Loading dataset

# In[127]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[96]:


df=pd.read_csv(r"C:\datasets\shoe_data.csv")


# # Data Preprocessing

# In[98]:


df


# In[100]:


print("Initial Data Shape:", df.shape)


# In[102]:


print(df.isnull().sum())


# In[130]:


df['avg_rating'].fillna(df['avg_rating'].median(), inplace=True)
df['description'].fillna('', inplace=True)


# # Generating TF-IDF Matrix

# In[135]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])


# # Computing Cosine Similarity

# In[74]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# # Product Index Mapping

# In[76]:


indices = pd.Series(df.index, index=df['name']).drop_duplicates()


# # Recomendation Function

# In[123]:


def get_recommendations(name, top_n=5):
    idx = indices.get(name)
    if idx is None:
        return f"Product '{name}' not found in the dataset."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # top 5 similar products
    shoe_indices = [i[0] for i in sim_scores]
    return df[['name','brand','price']].iloc[shoe_indices]
        


# # Testing Recomendation

# In[125]:


example_product = df.loc[0,'name']
print(f"Recommendations for: {example_product}\n")
recommendations = get_recommendations(example_product, top_n=5)
print(recommendations.to_string(index=False))


# ## visualisation of similiarity heat map of 20 products

# In[141]:


plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim[:20, :20], cmap='coolwarm')
plt.title("Cosine Similarity Matrix (Top 20 Products)")
plt.show()


# ## Accuracy Evaluation (Average top-5 similarity)

# In[146]:


avg_similarities = []
for i in range(len(cosine_sim)):
    sim_scores = cosine_sim[i]
    sim_scores[i] = 0  # exclude self
    avg_sim = np.mean(sorted(sim_scores, reverse=True)[:5])
    avg_similarities.append(avg_sim)
    


# In[148]:


print(f"\n Average top-5 similarity score across products: {np.mean(avg_similarities):.4f}")


# In[ ]:





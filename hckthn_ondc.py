# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA

# Data pipeline
def write_to_file(df):
   df.to_csv("/content/dummy_data.csv",mode='w',index=False)
   return

def positional_encoding(max_len, embedding_dim):
    position = np.arange(max_len, dtype=np.float32)
    angle_rates = 1 / np.power(10000, (2 * (np.arange(embedding_dim, dtype=np.float32) // 2)) / embedding_dim)
    angle_rads = np.expand_dims(position, -1) * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.expand_dims(angle_rads, 0)
    return pos_encoding

def index_to_embedding(index, pos_encoding):

    return pos_encoding[:, index]

def embedding_to_index(embedding, pos_encoding):
    # Calculate the cosine similarity between the embedding and each positional encoding
    similarity = np.sum(embedding * pos_encoding, axis=-1)
    # Find the index with the highest similarity (nearest neighbor search)
    index = np.argmax(similarity, axis=-1)
    return index

max_len = 256
embedding_dim = 4
vocab_size = 256

# Create sinusoidal positional encoding
pos_encoding = positional_encoding(max_len, embedding_dim)


"""data generation"""

btch=1000
n_chnk = 20

rows=0
for chunk in pd.read_csv("/content/dummy_data.csv", chunksize=1000):
  rows = rows + len(chunk)
n_chnk = (rows // btch ) + 1

# cluster suppliers or merchants by using Agglomerative clusering


# Initialize an empty DataFrame to store the aggregated data
aggregated_data = pd.DataFrame()

# Define the chunk size
chunk_size = 1000 ;
"""
threshold = 120.0 ; n_clusters= n_chnk  # Adjust as needed
clustering = AgglomerativeClustering(n_clusters=n_clusters)
# Iterate over each chunk in the dataset """
i=0
for chunk in pd.read_csv("/content/dummy_data.csv", chunksize=chunk_size):

    # Perform hierarchical clustering on the current chunk

    #Z = linkage(chunk, method='complete')

    # Apply flat clustering to assign each data point to a cluster
    # You can adjust the threshold to control the number of clusters

    #labels = clustering.fit_predict(chunk)

    # Add the cluster labels to the DataFrame
    chunk['cluster_label'] = i # labels
    i= i+1 

    # Save the aggregated data to a CSV file or perform further analysis
    chunk.to_csv("/content/cluster_data.csv",mode='a' ,index=False)



# Parameters
chunk_size = 1000  # Chunk size
filter_column = 'cluster_label'  # Column to filter by
filter_value = 1  # Value to filter on

# Read the CSV file in chunks
chunks = pd.read_csv("/content/cluster_data.csv", chunksize=chunk_size)

# Initialize an empty list to store filtered chunks
filtered_chunks = []

# Filter each chunk by the desired column label and value
for chunk in chunks:
    filtered_chunk = chunk[chunk[filter_column] == filter_value]
    filtered_chunks.append(filtered_chunk)
    filtered_chunk.to_csv("/content/filtered_data.csv",mode='a', index=False)

# Concatenate the filtered chunks into a single DataFrame
filtered_data = pd.concat(filtered_chunks)





# Parameters
variance_threshold = 0.9999  # Retain 100% of variance

# Filter each chunk by the desired column label and value
for chunk in  pd.read_csv("/content/filtered_data.csv", chunksize=300):
  clstr = chunk.drop(columns=['cluster_label']) ;
  btch = clstr.shape[0]
  X =  index_to_embedding(clstr.values.astype(int), pos_encoding)
  X = np.reshape(X , (btch,-1))

  # Initialize PCA with desired variance threshold
  pca = PCA(n_components=variance_threshold, svd_solver='full')
  # Fit PCA to the data
  X_pca = pca.fit_transform(X)

  # Number of principal components required to retain the specified variance threshold
  n_components_required = pca.n_components_

  # Total variance retained
  total_variance_retained = np.sum(pca.explained_variance_ratio_)

  # Print the number of principal components required and total variance retained
  print("Number of Principal Components Required:", n_components_required)
  print("Total Variance Retained:", total_variance_retained)

  # Initialize PCA with desired number of components
  n_components = n_components_required
  pca = PCA(n_components=n_components)

  # Fit PCA to the data and transform the data
  X_pca = pca.fit_transform(X)

  # Print original data shape and transformed data shape
  print("Original Data Shape:", X.shape)
  print("Transformed Data Shape:", X_pca.shape)

  # Print explained variance ratio
  #print("Explained Variance Ratio:", pca.explained_variance_ratio_)

  # Perform inverse transform
  X_inverse = pca.inverse_transform(X_pca)

  # Print data loss
  print("\nDifference between actual embedding and retrived embedding:")
  print(np.sum(np.absolute(X-X_inverse)))

  X_inverse = np.reshape(X_inverse, (btch,-1,1,4) )

  reconstructed_index = embedding_to_index(X_inverse, pos_encoding)
  reconstructed_index = np.reshape(reconstructed_index, (btch,-1))
  print("error on final data : ",np.sum(np.absolute(clstr.values-reconstructed_index)))

"""Converting label back to data"""

def dec2bin(dec):
  bin_str = format(int(dec),'b')
  return bin_str

binary_array = np.unpackbits(np.array(reconstructed_index, dtype=np.uint8), axis=1)
print(binary_array)

"""# conclusion

Actual size of data = 10M x 30K matrix

Proposed model
- array final storage = 10M x 300

- Overall compression ratio = 1 : 70

- Can be scaled similarly on to the rows to compress 10M rows

- Size of Look up matrix for embedding 256 x 4

- Size of cluster look up matrix = 1 x 10M

- Size of PCA object for each cluster = 0.35 KB

### Retrieval time

- time complexity = O(log n)

"""








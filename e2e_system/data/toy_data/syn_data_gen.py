import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_df(n_cont=2, n_cat=1, num_samples=10**6):
    # Create a dictionary with your columns
    data = dict()
    cont_names = []
    cat_names = []
    for i in range(n_cat):
        name = f'categorical_{i}'
        cat_names.append(name)
        data[name] = np.random.normal(loc=10, scale=1, size=num_samples).astype(int)
    for i in range(n_cont):
        name = f'continuous_{i}'
        cont_names.append(name)
        data[name] = np.random.normal(loc=100, scale=1, size=num_samples).astype(int)
      
    # Make sure the values are in the range you want
    for col in cat_names:
        data[col] = np.clip(data[col], 8, 11)
    for col in cont_names:
        data[col] = np.clip(data[col], 10, 1000)  
      
    # Create the dataframe  
    df = pd.DataFrame(data)
    
    return df

# set a seed for reproducibility
np.random.seed(0)

syn_df = create_df(n_cont=2, n_cat=1, num_samples=10000)
syn_df.to_csv('toy_syn_data.csv', index=False)

# read the data
syn_df_vae = pd.read_csv('../../syn_data/toy_syn_data.csv')
# plot a big figure, each column is a subplot
fig, axes = plt.subplots(nrows=1, ncols=syn_df_vae.shape[1], figsize=(20, 5))

for i, col in enumerate(syn_df_vae.columns):
    axes[i].hist(syn_df_vae[col], bins=50, color='blue', alpha=0.7)
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

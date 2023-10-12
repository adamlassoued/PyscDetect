import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder




# Centering the title using custom CSS
st.markdown(
    """
    <style>
        div.stTitle {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Med Fraud Shield")

# Description for the app
#st.write("Veuillez importer le relevé de l'année en cours.")



# Upload CSV file
uploaded_file = st.file_uploader("Importer les données", type=['xlsx', 'xls'], accept_multiple_files=False)


def perform_clustering_V1(data, sheet_name):
    df = pd.DataFrame(data)
    df['combined_feature'] = df['Medecin'].astype(str) + '_' + df['Medicament'].astype(str)

    # Selecting features for clustering
    features_for_clustering = df[['Medecin', 'Gouvernorat', 'Medicament', 'Nb_Ordonnance']]

    # Encode categorical features using LabelEncoder
    label_encoder = LabelEncoder()
    for col in ['Medecin', 'Gouvernorat', 'Medicament']:
        features_for_clustering[col] = label_encoder.fit_transform(features_for_clustering[col])

#KMEANS
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_for_clustering)

    cluster_mapping = df[['combined_feature', 'cluster']].drop_duplicates().set_index('combined_feature')['cluster'].to_dict()

    dfs = []
    for combined_feature, cluster_label in cluster_mapping.items():
        medecin, medicament = combined_feature.split('_')
        gouvernorat = df[df['combined_feature'] == combined_feature]['Gouvernorat'].iloc[0]  # Get Gouvernorat for the corresponding Medecin
        dfs.append(pd.DataFrame({'Medecin': [medecin], 'Gouvernorat': [gouvernorat], 'Medicament': [medicament], 'Cluster': [cluster_label]}))

    result_df = pd.concat(dfs, ignore_index=True)
    result_df['Sheet'] = sheet_name  # Add a column for sheet name
    return result_df



def perform_clustering_V2(data, sheet_name):
    df = pd.DataFrame(data)

    # Selecting features for clustering
    features_for_clustering = df[['Medecin', 'Gouvernorat', 'Medicament', 'Nb_Ordonnance']]

    # Encode categorical features using LabelEncoder
    label_encoder = LabelEncoder()
    for col in ['Medecin', 'Gouvernorat', 'Medicament']:
        features_for_clustering[col] = label_encoder.fit_transform(features_for_clustering[col])

    # Group by 'Medecin' and aggregate features
    medecin_features = features_for_clustering.groupby('Medecin').mean()

    # KMEANS
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    medecin_features['cluster'] = kmeans.fit_predict(medecin_features)

    # Convert the index of medecin_features to the same data type as 'Medecin' in df
    medecin_features.index = df['Medecin'].unique()

    # Merge cluster labels back to the original dataframe
    df = pd.merge(df, medecin_features[['cluster']], left_on='Medecin', right_index=True, how='left')

    result_df = df[['Medecin', 'Gouvernorat', 'cluster']].drop_duplicates()
    result_df.columns = ['Medecin', 'Gouvernorat', 'Cluster']
    result_df['Sheet'] = sheet_name  # Add a column for sheet name

    return result_df


if uploaded_file is not None:

    sheet_names = ['T1', 'T2', 'T3', 'T4']
    all_results = []

    for sheet_name in sheet_names:
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        result_df = perform_clustering_V2(data, sheet_name)
        all_results.append(result_df)


# Merge the results on "Medecin" column
    merged_df = all_results[0]
    for i in range(1, len(all_results)):
        merged_df = pd.merge(merged_df, all_results[i], on=['Medecin', 'Gouvernorat'], how='outer', suffixes=('', f'_sheet_{i+1}'))

# Create a final table with all four columns
    final_table = pd.DataFrame({
    'Medecin': merged_df['Medecin'],
    'Gouvernorat': merged_df['Gouvernorat'],
    'Trimestre 1': merged_df['Cluster'],
    'Trimestre 2': merged_df['Cluster_sheet_2'],
    'Trimestre 3': merged_df['Cluster_sheet_3'],
    'Trimestre 4': merged_df['Cluster_sheet_4']
})




# In[6]:


# Detection cluster change for each "Medecin"

# Create an empty list to store the results
    moved_medecins = []

# Iterate through the rows of the final_table DataFrame
    for index, row in final_table.iterrows():
    # Check if the cluster has moved from 0 to 2 in any trimester
        if (
        (row['Trimestre 1'] == 0 and row['Trimestre 2'] == 2) or
        (row['Trimestre 1'] == 0 and row['Trimestre 3'] == 2) or
        (row['Trimestre 1'] == 0 and row['Trimestre 4'] == 2) or
        (row['Trimestre 2'] == 0 and row['Trimestre 3'] == 2) or
        (row['Trimestre 2'] == 0 and row['Trimestre 4'] == 2) or
        (row['Trimestre 3'] == 0 and row['Trimestre 4'] == 2)
    ):
        # If the condition is met, add the "Medecin" to the list
            moved_medecins.append({
            'Medecin': row['Medecin'],
            'Gouvernorat': row['Gouvernorat'],
            'Trimestre 1': row['Trimestre 1'],
            'Trimestre 2': row['Trimestre 2'],
            'Trimestre 3': row['Trimestre 3'],
            'Trimestre 4': row['Trimestre 4'],
        })

# Create a DataFrame from the list of moved "Medecins"
    moved_medecins_df = pd.DataFrame(moved_medecins)

# Print the resulting DataFrame
    print("\nLes médecins qui ont fait 2 changements de classe:\n")
    print(moved_medecins_df)
    import streamlit as st


    st.markdown("\n**<span style='font-size:20px;'>Les médecins à activités suspectes :</span>**\n", unsafe_allow_html=True)
    st.dataframe(moved_medecins_df,width=800,hide_index=True)


# Load data for a specific trimester
    sheet_names = ['T1', 'T2', 'T3', 'T4']
    all_results = []

    for sheet_name in sheet_names:
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        result_df = perform_clustering_V1(data, sheet_name)
        all_results.append(result_df)

    
    # Merge the results on "Medecin" and "Medicament" columns
    merged_df = all_results[0]
    for i in range(1, len(all_results)):
        merged_df = pd.merge(merged_df, all_results[i], on=['Medecin', 'Medicament'], how='outer', suffixes=('', f'_sheet_{i+1}'))

# Create a final table with all four columns
    final_table = pd.DataFrame({
    'Medecin': merged_df['Medecin'],
    'Medicament': merged_df['Medicament'],
    'Trimestre 1': merged_df['Cluster'],
    'Trimestre 2': merged_df['Cluster_sheet_2'],
    'Trimestre 3': merged_df['Cluster_sheet_3'],
    'Trimestre 4': merged_df['Cluster_sheet_4']
})


# Detection cluster change

# Create an empty list to store the results
    moved_combinations = []

# Iterate through the rows of the final_tab
# le DataFrame
    for index, row in final_table.iterrows():
    # Check if the cluster has moved from 0 to 2 in any trimester
        if (
        (row['Trimestre 1'] == 0 and row['Trimestre 2'] == 2) or
        (row['Trimestre 1'] == 0 and row['Trimestre 3'] == 2) or
        (row['Trimestre 1'] == 0 and row['Trimestre 4'] == 2) or
        (row['Trimestre 2'] == 0 and row['Trimestre 3'] == 2) or
        (row['Trimestre 2'] == 0 and row['Trimestre 4'] == 2) or
        (row['Trimestre 3'] == 0 and row['Trimestre 4'] == 2)
    ):
        # If the condition is met, add the combination to the list
            moved_combinations.append({
            'Medecin': row['Medecin'],
            'Medicament': row['Medicament'],
            'Trimestre 1': row['Trimestre 1'],
            'Trimestre 2': row['Trimestre 2'],
            'Trimestre 3': row['Trimestre 3'],
            'Trimestre 4': row['Trimestre 4'],
        })

# Create a DataFrame from the list of moved combinations
    moved_combinations_df = pd.DataFrame(moved_combinations)

# Print the resulting DataFrame
    print("\nLes combinaisons médecin/médicament qui ont fait 2 changements de classe:\n")
    print(moved_combinations_df)
    st.markdown("\n**<span style='font-size:20px;'>Le comportement suspects des médecins par médicament :</span>**\n", unsafe_allow_html=True)
    st.dataframe(moved_combinations_df, width=800,hide_index=True)

# In[7]:


# Save the DataFrame to a CSV file
    moved_combinations_df.to_csv('medecins_medicaments_detectes.csv', index=False)
    
    # # Download the processed data as a CSV file
    # result_csv = result_df.to_csv(index=False)
    # st.download_button(
    #     label="Download Processed CSV",
    #     data=result_csv.encode('utf-8'),
    #     key='download_button',
    #     file_name='processed_data.csv',
    #     mime='text/csv'
    # )

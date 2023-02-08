import urllib.request
import json
import os
import argparse
from pathlib import Path
import pandas as pd

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument('--output', type=str)


args = parser.parse_args()



 

output_path = f"{Path(args.output)}/output/"
print(f"output_path{output_path}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

path = f"{Path(args.input)}"


### load csv file
df_in = pd.read_csv(f"{path}/clustering.csv")  ## or utf-8

## add a column of product id
df_in['Product_ID'] = range(1,df_in.shape[0]+1)
df_in['BestPrice'] = 'No'
df_in['Cluster_Average'] = 0
print('file overview')
print(df_in.head())

df = df_in.loc[df_in['Amount'] !=  None]

df['file_name'] = [item.replace('__','/') for item in df['file_name']]
df['Bing_Search_Description'] = [str(item).replace('<b>',' ') for item in df['Bing_Search_Description']]
df['Bing_Search_Description'] = [str(item).replace('</b>',' ') for item in df['Bing_Search_Description']]
df['Bing_Entity_Description'] = [str(item).replace('<b>',' ') for item in df['Bing_Entity_Description']]
df['Bing_Entity_Description'] = [str(item).replace('</b>',' ') for item in df['Bing_Entity_Description']]



print(df.head())
 
for j in range (1,max(list(df.loc[df['TypeClass'] ==  'C']['ClusterID']))+1):
    sub_df = df.loc[df['ClusterID'] == j]
    sub_df.reset_index(inplace = True, drop = True)


    a_list = list(sub_df['Amount'])
    row_id = sub_df
    min_index = a_list.index(min(a_list))
    
    row_id = sub_df.iloc[[min_index]]['RowID']
    try:
        avg_amount =  round(sum(a_list) / len(a_list),2 )
    except:
        avg_amount = 0
    
    df.loc[df['RowID'] == int(row_id),['BestPrice']] = 'Yes'
    df.loc[df['ClusterID'] == j, ['Cluster_Average']] = avg_amount
 
df.to_excel(f"{output_path}clustering.xlsx")





account_name = '<add Azure storage name>'
account_key = '<add Azure storage key>'
container_name = '<add Azure storage container key>'
 
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"


blob_svc = BlobServiceClient.from_connection_string(conn_str=connection_string)

local_file_name = "clustering.xlsx"

blob_client = blob_svc.get_blob_client(container=container_name, blob=local_file_name)

with open(file=f"{output_path}clustering.xlsx", mode="rb") as data:
    blob_client.upload_blob(data, overwrite=True)

print("Uploading Complete")

             
print(f"data preparation for visulization process complete")

 
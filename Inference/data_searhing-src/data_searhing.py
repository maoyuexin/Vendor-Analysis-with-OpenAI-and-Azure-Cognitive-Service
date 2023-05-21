import urllib.request
import json
import os
import argparse
from pathlib import Path
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient



parser = argparse.ArgumentParser()
parser.add_argument("--file_path_in", type=str)
parser.add_argument('--file_path_out', type=str)


 
search_creds = "<Add Search Credential>"
searchservice = "<Add Search Key>"
search_creds = AzureKeyCredential(search_creds)




args = parser.parse_args()

path = f"{Path(args.file_path_in)}/categorized_json/"

output_path = f"{Path(args.file_path_out)}/cog_search_json/"
print(f"output_path{output_path}")
if not os.path.exists(output_path):
    os.makedirs(output_path)



files = os.listdir(path)
cs_index = 'product-invoice'

search_client = SearchClient(endpoint=f"https://{searchservice}.search.windows.net/",
                      index_name= cs_index,
                      credential= search_creds)


for fileName in files:
    with open(f"{path}{fileName}", 'rb') as f2: 
        json_results = json.load(f2)

        for idx, invoice in enumerate(json_results['documents']): 
            vendorName = invoice.get("VendorName")

            for idx2, product in enumerate(json_results['documents'][idx]['Items']): 
                Description = product.get("Product_Summarization")

                if Description =='':
                    Description = product.get("Description")
                
                ### Search form Cog Search and get top N similar products
                cog_results = search_client.search(search_text=Description, top = 3, include_total_count=True)
                search_results = []
                for search_result in cog_results:
                    search_results.append(search_result)

                json_results['documents'][idx]['Items'][idx2]['search_results'] = search_results
    
    with open(f"{output_path}{fileName}", 'w') as f3: 
            json.dump(json_results, f3)

print(f" data search process complete")
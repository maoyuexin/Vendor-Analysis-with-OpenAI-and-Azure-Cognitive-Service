# import libraries
import requests

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import os
import json
import argparse
from pathlib import Path
global language_key  
global bing_key  


def bing_search_entities(query):
    
    search_url = "https://api.bing.microsoft.com/v7.0/entities"
    mkt = 'en-US'

    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    #params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    params = {"q": query, "mkt": mkt}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return(search_results)



def bing_search(query):
     
    search_url = "https://api.bing.microsoft.com/v7.0/search"
 
    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
     

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return(search_results)


def extract_json_bing_search(search_result):
    try:
        searchResults = search_result.get('webPages').get('value')
        listDescription = []
        listKeyPhase = []
        for searchItem in searchResults:
            
            description = searchItem.get("snippet")
            listDescription.append(description)
            try:
                
                keyPhase = extract_key_phrases([description])[0]['key_phrases']
                listKeyPhase.append(' '.join(keyPhase))
            except Exception as e:
                listKeyPhase.append(None)


        rst = [' '.join(listDescription), ' '.join(listKeyPhase)]
    except Exception as e:
        rst = [None, None]
    
    return(rst)

 
def extract_key_phrases(articles):

    endpoint = 'https://language-service-for-custom-text.cognitiveservices.azure.com/'
    
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(language_key))
    result = text_analytics_client.extract_key_phrases(articles)

    return(result)


def extract_json_bing_entity(entity_result):
    try:
        entities = entity_result.get('entities').get('value')
        listEntityType = []
        listDescription = []
        listKeyPhase=[]
        for entityItem in entities:
            if entityItem.get("entityPresentationInfo").get("entityTypeHints"):
                entityType = ' '.join(entityItem.get("entityPresentationInfo").get("entityTypeHints"))
                listEntityType.append(entityType)
            else:
                entityType = None
            description = entityItem.get("description")

            ## extract key phase
            try:
                listKeyPhase.append(' '.join(extract_key_phrases([description])[0]['key_phrases']))
            except Exception as e:
                listKeyPhase.append(None)

            listDescription.append(description)
        rst = [' '.join(listEntityType), ' '.join(listDescription), ' '.join(listKeyPhase)]


    except Exception as e:
        rst = [None, None, None]
    
    return(rst)


parser = argparse.ArgumentParser()
parser.add_argument("--file_path_in", type=str)
parser.add_argument('--file_path_out', type=str)


args = parser.parse_args()


bing_key = "<add bing key>"
language_key = '<add cognitive service key>'

 

output_path = f"{Path(args.file_path_out)}/enriched_json/"
print(f"output_path{output_path}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

path = f"{Path(args.file_path_in)}/raw_json/"

files = os.listdir(path)
print(files)

for fileName in files:
    if fileName not in ["tracker.csv"]:
        print(f"{path}{fileName}")
        with open(f"{path}{fileName}", 'rb') as f2: 
            json_results = json.load(f2)
        print(json_results)
        for idx, invoice in enumerate(json_results['documents']): 
            vendor_name = invoice.get("VendorName")
            if vendor_name:
                bing_search_results = extract_json_bing_search(bing_search(vendor_name))
                
                print(f"Bing Search Results: {bing_search_results}")
                json_results['documents'][idx]['Bing_Search_Description'] = bing_search_results[0]
                json_results['documents'][idx]['Bing_Search_KeyPhase'] = bing_search_results[1]

                bing_entity_results= extract_json_bing_entity(bing_search_entities(vendor_name))
                json_results['documents'][idx]['Bing_Entity_Type'] = bing_entity_results[0]
                json_results['documents'][idx]['Bing_Entity_Description'] = bing_entity_results[1]
                json_results['documents'][idx]['Bing_Entity_KeyPhase'] = bing_entity_results[2]
        
        with open(f"{output_path}{fileName}", 'w') as f3: 
            json.dump(json_results, f3)

print(f"data enrichment process complete")
import urllib.request
import json
import os
import argparse
from pathlib import Path
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

import pymsteams




def create_team_card(fileName, Webhook_URL, document, product_description, product_summary, product_price, vendor_name, best_price, savings ):


    myTeamsMessage = pymsteams.connectorcard(Webhook_URL)

    myTeamsMessage.text(f"Invoice Report: {fileName.replace('.json','')}")

    # create the section
    myMessageSection = pymsteams.cardsection()

    # Section Title
    myMessageSection.title("Product Infomration:")
    myMessageSection.addFact("Product Description", product_description)
    myMessageSection.addFact("AOAI Summary", product_summary)
    myMessageSection.addFact("Vendor Name", vendor_name)
    myMessageSection.addFact("Price", product_price)
     

    myMessageSection2 = pymsteams.cardsection()

    myMessageSection2.title("Similar Products in the Product Management System:")
    
    myMessageSection2.activityTitle(product['search_results'][0]['Product_Description'])
    myMessageSection2.activitySubtitle(product['search_results'][0]['VendorName'])
    myMessageSection2.activityText(f"Price:{product['search_results'][0]['Amount']}")

    myMessageSection3 = pymsteams.cardsection()
    myMessageSection3.activityTitle(product['search_results'][1]['Product_Description'])
    myMessageSection3.activitySubtitle(product['search_results'][1]['VendorName'])
    myMessageSection3.activityText(f"Price:{product['search_results'][1]['Amount']}")

    myMessageSection4 = pymsteams.cardsection()
    myMessageSection4.activityTitle(product['search_results'][1]['Product_Description'])
    myMessageSection4.activitySubtitle(product['search_results'][1]['VendorName'])
    myMessageSection4.activityText(f"Price: {product['search_results'][1]['Amount']}")
 

    myMessageSection5 = pymsteams.cardsection()
    myMessageSection5.title("Summary:")
    myMessageSection5.addFact("Product has best Price: ", best_price)
    myMessageSection5.addFact("Product Price Saving: ", savings)


    
    # Add your section to the connector card object before sending
    myTeamsMessage.addSection(myMessageSection)
    myTeamsMessage.addSection(myMessageSection2)
    myTeamsMessage.addSection(myMessageSection3)
    myTeamsMessage.addSection(myMessageSection4)
    myTeamsMessage.addSection(myMessageSection5)

    myTeamsMessage.send()




Webhook_URL  = '<Add Web hook URL>'

parser = argparse.ArgumentParser()
parser.add_argument("--file_path_in", type=str)
parser.add_argument('--file_path_out', type=str)





args = parser.parse_args()

path = f"{Path(args.file_path_in)}/cog_search_json/"

output_path = f"{Path(args.file_path_out)}/output/"
print(f"output_path{output_path}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

best_price = 'No'
savings = '-40%'

files = os.listdir(path)


for fileName in files:
    with open(f"{path}{fileName}", 'rb') as f2: 
        json_results = json.load(f2)

        for idx, invoice in enumerate(json_results['documents']): 
            vendorName = invoice.get("VendorName")

            for idx2, product in enumerate(json_results['documents'][idx]['Items']): 
                
                Description = product.get("Description")
                Summary_Description = product.get("Product_Summarization")
                Price = product.get("Amount")
                
                group_id = product['search_results'][0]['Group_Id']
                create_team_card(fileName, Webhook_URL, product, Description, Summary_Description, Price, vendorName, best_price, savings)


    with open(f"{output_path}{fileName}", 'w') as f3: 
            json.dump(json_results, f3)

print(f"Sent Message to teams")
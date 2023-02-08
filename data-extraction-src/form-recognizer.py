# import libraries
import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from datetime import datetime, timedelta
from azure.storage.blob import BlobClient, generate_blob_sas, BlobSasPermissions
import pickle
import json
import pandas as pd
import argparse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from pathlib import Path

import re


def remove_non_ascii(term):
    term = str(term)
    term = term.replace("\n", " ")
    return re.sub(r'[^\x00-\x7f]',"", term)

def remove_leading_trailing_specialChar(term):
    term = str(term)
    term = term.strip()
    term = term.strip('[@_!#$%^&*()<>?/\|}{~:].,')
    return term



def format_bounding_region(bounding_regions):
    if not bounding_regions:
        return "N/A"
    return ", ".join("Page #{}: {}".format(region.page_number, format_polygon(region.polygon)) for region in bounding_regions)

def format_polygon(polygon):
    if not polygon:
        return "N/A"
    return ", ".join(["[{}, {}]".format(p.x, p.y) for p in polygon])


def analyze_invoice(invoices, file_name):

    dict_invoice = {"file_name": file_name, "documents":[]}

    for idx, invoice in enumerate(invoices.documents):
        document_field = {}
        #print("--------Recognizing invoice #{}--------".format(idx + 1))
        vendor_name = invoice.fields.get("VendorName")
        if vendor_name:
            print(f"original custoemr name: {vendor_name.value}")
            vendor_clean = remove_leading_trailing_specialChar(remove_non_ascii(vendor_name.value))
            print(f"updated custoemr name: {vendor_clean}")
            document_field['VendorName'] = vendor_clean
            document_field['VendorName_conf'] = vendor_name.confidence
             
        vendor_address = invoice.fields.get("VendorAddress")
 
        if vendor_address:
            document_field['VendorAddress'] = {'house_number':vendor_address.value.house_number,'po_box':vendor_address.value.po_box, 
            'road':vendor_address.value.road, 'city':vendor_address.value.city,'state':vendor_address.value.state,
            'postal_code':vendor_address.value.postal_code,'country_region':vendor_address.value.country_region,'street_address':vendor_address.value.street_address,
            }
            document_field['VendorAddress_conf'] = vendor_address.confidence
            
        vendor_address_recipient = invoice.fields.get("VendorAddressRecipient")
        if vendor_address_recipient:
            document_field['VendorAddressRecipient'] = vendor_address_recipient.value
            document_field['VendorAddressRecipient_conf'] = vendor_address_recipient.confidence
       
        customer_name = invoice.fields.get("CustomerName")
        if customer_name:
            print(f"original custoemr name: {customer_name.value}")
            customer_des_clean = remove_leading_trailing_specialChar(remove_non_ascii(customer_name.value))
            print(f"updated custoemr name: {customer_des_clean}")
            document_field['CustomerName'] = customer_des_clean
            document_field['CustomerName_conf'] = customer_name.confidence
            
           
        customer_id = invoice.fields.get("CustomerId")
        if customer_id:
            document_field['CustomerId'] = customer_id.value
            document_field['CustomerId_conf'] = customer_id.confidence
        
        customer_address = invoice.fields.get("CustomerAddress")
        if customer_address:
            document_field['CustomerAddress'] = {'house_number':customer_address.value.house_number,'po_box':customer_address.value.po_box, 
            'road':customer_address.value.road, 'city':customer_address.value.city,'state':customer_address.value.state,
            'postal_code':customer_address.value.postal_code,'country_region':customer_address.value.country_region,'street_address':customer_address.value.street_address,
            }
            document_field['CustomerAddress_conf'] = customer_address.confidence

        customer_address_recipient = invoice.fields.get("CustomerAddressRecipient")
        if customer_address_recipient:
            document_field['CustomerAddressRecipient'] = customer_address_recipient.value
            document_field['CustomerAddressRecipient_conf'] = customer_address_recipient.confidence
         
        invoice_id = invoice.fields.get("InvoiceId")
        if invoice_id:
            document_field['InvoiceId'] = invoice_id.value
            document_field['InvoiceId_conf'] = invoice_id.confidence

        invoice_date = invoice.fields.get("InvoiceDate")
        if invoice_date:
            document_field['InvoiceDate'] = str(invoice_date.value)
            document_field['InvoiceDate_conf'] = invoice_date.confidence

        invoice_total = invoice.fields.get("InvoiceTotal")
        if invoice_total:
            document_field['InvoiceTotal'] = invoice_total.value.amount
            document_field['InvoiceTotal_conf'] = invoice_total.confidence
         
        due_date = invoice.fields.get("DueDate")
        if due_date:
            document_field['DueDate'] = str(due_date.value)
            document_field['DueDate_conf'] = due_date.confidence

        purchase_order = invoice.fields.get("PurchaseOrder")
        if purchase_order:
            document_field['PurchaseOrder'] = purchase_order.value
            document_field['PurchaseOrder_conf'] = purchase_order.confidence
         
        billing_address = invoice.fields.get("BillingAddress")
        if billing_address:
            document_field['BillingAddress'] = {'house_number':billing_address.value.house_number,'po_box':billing_address.value.po_box, 
            'road':billing_address.value.road, 'city':billing_address.value.city,'state':billing_address.value.state,
            'postal_code':billing_address.value.postal_code,'country_region':billing_address.value.country_region,'street_address':billing_address.value.street_address,
            }
            document_field['BillingAddress_conf'] = billing_address.confidence

        billing_address_recipient = invoice.fields.get("BillingAddressRecipient")
        if billing_address_recipient:
            document_field['BillingAddressRecipient'] = billing_address_recipient.value
            document_field['BillingAddressRecipient_conf'] = billing_address_recipient.confidence

        shipping_address = invoice.fields.get("ShippingAddress")
        if shipping_address:
            document_field['ShippingAddress'] = {'house_number':shipping_address.value.house_number,'po_box':shipping_address.value.po_box, 
            'road':shipping_address.value.road, 'city':shipping_address.value.city,'state':shipping_address.value.state,
            'postal_code':shipping_address.value.postal_code,'country_region':shipping_address.value.country_region,'street_address':shipping_address.value.street_address,
            }
            document_field['ShippingAddress_conf'] = shipping_address.confidence

        shipping_address_recipient = invoice.fields.get("ShippingAddressRecipient")
        if shipping_address_recipient:
            document_field['ShippingAddressRecipient'] = shipping_address_recipient.value
            document_field['ShippingAddressRecipient_conf'] = shipping_address_recipient.confidence

        #print("Invoice items:")
        items = []
        get_items = invoice.fields.get("Items")
        if get_items:
            for idx, item in enumerate(get_items.value):
                #print("...Item #{}".format(idx + 1))
                item_dict = {}
                item_description = item.value.get("Description")
                if item_description:
                    print(f"original item desciption: {item_description.value}")
                    item_des_clean = remove_leading_trailing_specialChar(remove_non_ascii(item_description.value))
                    print(f"updated item desciption: {item_des_clean}")
                    item_dict['Description'] = item_des_clean
                    item_dict['Description_conf'] = item_description.confidence

                item_quantity = item.value.get("Quantity")
                if item_quantity:
                    item_dict['Quantity'] = item_quantity.value
                    item_dict['Quantity_conf'] = item_quantity.confidence

                unit = item.value.get("Unit")
                if unit:
                    item_dict['Unit'] = unit.value
                    item_dict['Unit_conf'] = unit.confidence

                unit_price = item.value.get("UnitPrice")
                if unit_price:
                    item_dict['UnitPrice'] = unit_price.value.amount
                    item_dict['UnitPrice_conf'] = unit_price.confidence

                product_code = item.value.get("ProductCode")
                if product_code:
                    item_dict['ProductCode'] = product_code.value
                    item_dict['ProductCode_conf'] = product_code.confidence

                item_date = item.value.get("Date")
                if item_date:
                    item_dict['Date'] = str(item_date.value)
                    item_dict['Date_conf'] = item_date.confidence
    
                tax = item.value.get("Tax")
                if tax:
                    item_dict['Tax'] = tax.value.amount
                    item_dict['Tax_conf'] = tax.confidence

                amount = item.value.get("Amount")
                if amount:
                    item_dict['Amount'] = amount.value.amount
                    item_dict['Amount_conf'] = amount.confidence
            
                items.append(item_dict)

        document_field['Items'] = items
        
        subtotal = invoice.fields.get("SubTotal")
        if subtotal:
            document_field['SubTotal'] = subtotal.value.amount
            document_field['SubTotal_conf'] = subtotal.confidence

        total_tax = invoice.fields.get("TotalTax")
        if total_tax:
            document_field['TotalTax'] = total_tax.value.amount
            document_field['TotalTax_conf'] = total_tax.confidence

        previous_unpaid_balance = invoice.fields.get("PreviousUnpaidBalance")
        if previous_unpaid_balance:
            document_field['PreviousUnpaidBalance'] = previous_unpaid_balance.value.amount
            document_field['PreviousUnpaidBalance_conf'] = previous_unpaid_balance.confidence

        amount_due = invoice.fields.get("AmountDue")
        if amount_due:
            document_field['AmountDue'] = amount_due.value.amount
            document_field['AmountDue_conf'] = amount_due.confidence

        service_start_date = invoice.fields.get("ServiceStartDate")
        if service_start_date:
            document_field['ServiceStartDate'] = str(service_start_date.value)
            document_field['ServiceStartDate_conf'] = service_start_date.confidence

        service_end_date = invoice.fields.get("ServiceEndDate")
        if service_end_date:
            document_field['ServiceEndDate'] = str(service_end_date.value)
            document_field['ServiceEndDate_conf'] = service_end_date.confidence

        service_address = invoice.fields.get("ServiceAddress")
        if service_address:
            document_field['ServiceAddress'] = {'house_number':service_address.value.house_number,'po_box':service_address.value.po_box, 
            'road':service_address.value.road, 'city':service_address.value.city,'state':service_address.value.state,
            'postal_code':service_address.value.postal_code,'country_region':service_address.value.country_region,'street_address':service_address.value.street_address,
            }
            document_field['ServiceAddress_conf'] = service_address.confidence

        service_address_recipient = invoice.fields.get("ServiceAddressRecipient")
        if service_address_recipient:
            document_field['ServiceAddressRecipient'] = service_address_recipient.value
            document_field['ServiceAddressRecipient_conf'] = service_address_recipient.confidence

        remittance_address = invoice.fields.get("RemittanceAddress")
        if remittance_address:
            document_field['RemittanceAddress'] = {'house_number':remittance_address.value.house_number,'po_box':remittance_address.value.po_box, 
            'road':remittance_address.value.road, 'city':remittance_address.value.city,'state':remittance_address.value.state,
            'postal_code':remittance_address.value.postal_code,'country_region':remittance_address.value.country_region,'street_address':remittance_address.value.street_address,
            }
            document_field['RemittanceAddress_conf'] = remittance_address.confidence

        remittance_address_recipient = invoice.fields.get("RemittanceAddressRecipient")
        if remittance_address_recipient:
            document_field['RemittanceAddressRecipient'] = remittance_address_recipient.value
            document_field['RemittanceAddressRecipient_conf'] = remittance_address_recipient.confidence
    
    dict_invoice["documents"].append(document_field)
    #print("----------------------------------------")
    return dict_invoice


def get_blob_sas(account_name,account_key, container_name, blob_name):
    sas_blob = generate_blob_sas(account_name=account_name, 
                                container_name=container_name,
                                blob_name=blob_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(hours=1))
    return sas_blob

parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int)
parser.add_argument("--end_index", type=int)
parser.add_argument("--output", type=str)

args = parser.parse_args()



account_name = '<add Azure storage name>'
account_key = '<add Azure storage key>'
container_name = '<add Azure storage container key>'
endpoint = '<add form recognizer endpoint>'
key = '<add form recognizer key>'
start_index = args.start_index
end_index = args.end_index

connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
blob_svc = BlobServiceClient.from_connection_string(conn_str=connection_string)


container_client = blob_svc.get_container_client(container_name)

blob_metatdata_path = "<blob_metatdata_path point to invoice metadata csv file>"
blob_client = container_client.get_blob_client(blob_metatdata_path)

print(blob_client)

download_stream = blob_client.download_blob()

with open("invoice_metadata.csv", "wb") as f:
   download_stream.readinto(f)

df = pd.read_csv("invoice_metadata.csv",header=None, names = ['path'], index_col=False)
print(df)


output_path = f"{Path(args.output)}/raw_json/"
if not os.path.exists(output_path):
    os.makedirs(output_path)



document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

tracker = {'files':[], 'status':[]}
for index, item in enumerate(df['path'][start_index:end_index]):

    try:
        print(f"Processing the {index} document:")
        blob_name = "rvl-cdip/images/" + item
         
        blob_sas = get_blob_sas(account_name,account_key, container_name, blob_name)
        url = 'https://'+account_name+'.blob.core.windows.net/'+container_name+'/'+blob_name+'?'+blob_sas
        print(item)
        print(url)
        #invoices= analyze_invoice(url, endpoint, key)   

        poller = document_analysis_client.begin_analyze_document_from_url(
                "prebuilt-invoice", url)
        invoices = poller.result()

        
        ### Save the raw file in pickle  -- have issues to directly save as JSON file
        file_name = item.replace('/','__')
        #invoice_dict = invoices.to_dict()
        #with open(f"poc_results/raw/{file_name}.pickle", 'wb') as f: 
        #    pickle.dump(invoices, f )

        ### Save the extracted files into JSON  
        extraced_json = analyze_invoice(invoices, file_name)
        print(extraced_json)

        with open(f"{output_path}/{file_name}.json", 'w') as f2: 
            json.dump(extraced_json, f2)


        tracker['files'].append(item)
        tracker['status'].append(1)
        #tracker['msg'].append('')
        print(f"The {index} document completed:")
    except Exception as e:
        tracker['files'].append(item)
        tracker['status'].append(0)
        #tracker['msg'].append(str(e))

        print(f"The {index} document failed:")
        pass

result_tracker = pd.DataFrame.from_dict(tracker)
result_tracker.to_csv(f"{Path(args.output)}/tracker.csv", index = False)
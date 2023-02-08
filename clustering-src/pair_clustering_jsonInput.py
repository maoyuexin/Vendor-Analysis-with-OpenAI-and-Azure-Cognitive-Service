import pandas as pd
import recordlinkage
from recordlinkage.index import Block, Full
from itertools import chain
import json
from timeit import default_timer as timer
import numpy as np
import os
import argparse
from pathlib import Path
import copy

comparison_types = {'object': 'fuzzy', 'int64': 'exact', 'float64': 'exact', 'datetime64': 'exact', 'bool': 'exact'}
DEFAULT_RL_MODEL = 'ECM Classifier'

RECORDLINKAGE_ALGORITHMS = [
	{"name": "ECM Classifier", 'hyperParameter': {"init": "jaro", "binarize": 0.8, "max_iter": 100}},
	{"name": "K-Means Classifier", 'hyperParameter': {"match_cluster_center": None, "non-match_cluster_center": None}}]
SUPERVIZED_ALG_UPPER = 0.9
SUPERVIZED_ALG_LOWER = 0.7
clusteirng_algorithm = 'ecm'


def get_input_data(filePath, fileName, fileType):
    try:
        ## read file
         
        if fileType == 'pkl':
            df = pd.read_pickle(f"{filePath}{fileName}.pkl")
            types = df.dtypes.to_dict()

        elif fileType == 'csv':
            df = pd.read_csv(f"{filePath}{fileName}.csv", encoding='latin')  ## or utf-8
            types = df.dtypes.to_dict()
        
        elif fileType =='blob_csv':
            df = pd.read_csv(filePath, encoding='latin')  ## or utf-8
            types = df.dtypes.to_dict()

        else:
            return {'status': False, 'err_msg': f'[ERROR: {fileType} is not supported]'}

        return {'status': True, 'data': df, 'metadata': types, 'msg': f"[INFO: Read Input File]"}

    except Exception as e:
        return {'status': False, 'err_msg': '[ERROR: ' + str(e) + ']'}   



def set_match_attributes(match_parameters, field_types):
    global comparison_types
 
    try:
        match_fields = []
        if all(isinstance(item, str) for item in match_parameters):

            # Handles EM Inputs without comparison type info; using mapping from config
            match_fields = [{'attribute_name': item, "comparison_type": comparison_types[field_types[item].name]}
                            for item in match_parameters]
            return {'status': True, 'data': match_fields}

        elif all(isinstance(item, dict) for item in match_parameters):

            # Regular cases where comparison types are provided
            if all('comparison_type' in item for item in match_parameters):
                return {'status': True, 'data': match_parameters}

            # Comparison cases without comparison type info; using mapping from config (Weighted cases)
            else:
                match_fields = match_parameters
                for item in match_fields:
                    item['comparison_type'] = comparison_types[field_types[item['attribute_name']].name]
                return {'status': True, 'data': match_fields}
        else:
            return {'status': False, 'data': None,
                    'err_msg': '[Error: set_match_attributes() failed with Unexpected DataType for match fields]'}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}', 'data': None}




def set_comparison_parameters(match_fields, candidate_links, original_df):

    #Defaults
    string_method = 'jarowinkler'
    string_threshold = None

    numeric_method = 'gauss'
    offset = 0.0
    scale = 1.0
    origin = 0.0

    missing_value = 0

    try:
        comparer = recordlinkage.Compare()

        for field in match_fields:

            attribute = field['attribute_name']
            match_type = field['comparison_type']

            if match_type == 'exact':
                e_missing_value = field['missing_value'] if 'missing_value' in field else missing_value

                comparer.exact(attribute, attribute, label=attribute, missing_value=e_missing_value)

            elif match_type == 'numeric':
                method = field['method'] if 'method' in field else numeric_method
                n_offset = field['offset'] if 'offset' in field else offset
                n_scale = field['scale'] if 'scale' in field else scale
                n_origin = field['origin'] if 'origin' in field else origin
                n_missing_value = float(field['missing_value'] if 'missing_value' in field else missing_value)

                comparer.numeric(attribute, attribute, label=attribute, method=method, offset=n_offset, scale=n_scale,
                                 origin=n_origin, missing_value=n_missing_value)

            elif match_type == 'fuzzy':
                method = field['method'] if 'method' in field else string_method
                threshold = field['threshold'] if 'threshold' in field else string_threshold
                s_missing_value = float(field['missing_value'] if 'missing_value' in field else missing_value)

                comparer.string(attribute, attribute, label=attribute, method=method, threshold=threshold,
                                missing_value=s_missing_value)

            else:
                return {'status': False, 'data': None,
                        'err_msg': f'[ERROR: "{match_type}" match type is not currently supported]'}

        #Create & Return Feature Vector
        features = comparer.compute(candidate_links, original_df)
        return {'status': True, 'data': features, 'msg': f"[INFO: Generated Feature Vector]"}

    except Exception as e:
        return {'status': False, 'data': None, 'err_msg': '[ERROR: ' + str(e) + ']'}


def weighted_score_process(comparison_vector, match_fields, s_threshold):
    try:
        modified_vector = comparison_vector.copy()

        ### Generate Weighted Scores
        for field in match_fields:
            modified_vector[field['attribute_name']] = modified_vector[field['attribute_name']] * field['weight']

        ### Identify Matches based on Similarity Threshold
        all_sim_scores = modified_vector.sum(axis=1)
        matches = modified_vector[all_sim_scores > s_threshold]

        ### Generate Multi-Index pair for matches
        index_1 = list(matches.index.get_level_values(level=0))
        index_2 = list(matches.index.get_level_values(level=1))
        tuples_list = list(zip(index_1, index_2))

        #return {'status': True, 'data': {'feature_vector': modified_vector,'similarity_scores': all_sim_scores, 'match_pairs': tuples_list}}
        return {'status': True, 'data': {'similarity_scores': all_sim_scores, 'match_pairs': tuples_list}}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}', 'data': None}

def recordlinkage_model_fitting(model, comparison_vector):
    try:
        model.fit(comparison_vector)
        return {'status': True, 'data': {'model': model}}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}]'}


def recordlinkage_model_prediction(model, comparison_vector):
    try:
        result = model.predict(comparison_vector)
        return {'status': True, 'data': {'model_prediction': result}}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}'}


def recordlinkage_model_evaluation(model):

    try:
        model_evl = {"p probability P(Match)": model.p,
                     "m probabilities P(x_i=1|Match)": model.m_probs,
                     "u probabilities P(x_i=1|Non-Match)": model.u_probs,
                     "log m probabilities P(x_i=1|Match)": model.log_m_probs,
                     "log u probabilities P(x_i=1|Non-Match)": model.log_u_probs,
                     "log weights of features": model.log_weights,
                     "weights of features": model.weights
                     }

        data = {"Model Evaluation": model_evl}

        return {'status': True, 'data': data, 'msg': 'Model Evaluation Parameters generated'}

    except Exception as e:
        return {'status': False, 'err_msg': str(e), 'data': None}


def recordlinkage_classifier_process(features, model_parameters):
    global DEFAULT_RL_MODEL
    global RECORDLINKAGE_ALGORITHMS

    try:

        model_name = DEFAULT_RL_MODEL
        model = None

        if 'model_name' in model_parameters:
            model_name = model_parameters['model_name']

        #Get Model HyperParameters
        hyperParameters = [item['hyperParameter'] for item in RECORDLINKAGE_ALGORITHMS if item['name'] == model_name][0]

        #Initialize the Model
        if model_name == "ECM Classifier":
            init_method = model_parameters['method'] if 'method' in model_parameters else hyperParameters['init']
            binarize = model_parameters['binary_threshold'] if 'binary_threshold' in model_parameters else hyperParameters['binarize']
            iter = model_parameters['max_iters'] if 'max_iters' in model_parameters else hyperParameters['max_iter']
            model = recordlinkage.ECMClassifier(init=init_method, binarize=binarize, max_iter=iter)

        #Model Fitting
        fit_rst = recordlinkage_model_fitting(model, features)

        if not fit_rst['status']:
            return {'status': False, 'err_msg': f"[Error: recordlinkage_model_fitting() failed with {fit_rst['err_msg']}"}

        fit_model = fit_rst['data']['model']

        #Match Probabilities
        # TODO: Can get this done using a separate method and same can be done for model evaluation
        probability = fit_model.prob(features)

        #Model Prediction
        pred_rst = recordlinkage_model_prediction(fit_model, features)
        if not pred_rst['status']:
            return {'status': False, 'err_msg': f"[Error: recordlinkage_model_prediction() failed with {pred_rst['err_msg']}"}

        matches = pred_rst['data']['model_prediction']
        #print(matches)
        #Model Evaluation
        eval_rst = recordlinkage_model_evaluation(fit_model)
        if not eval_rst['status']:
            return {'status': False,
                    'err_msg': f"[Error: recordlinkage_model_evaluation() failed with {eval_rst['err_msg']}"}

        model_eval = eval_rst['data']

        return {'status': True, 'data': {'prediction': list(matches.values), 'match_probability': probability,
                'evaluation': model_eval}}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}', 'data': None}



def generate_training_set(feature_vector, probabilities):
    global SUPERVIZED_ALG_UPPER 
    global SUPERVIZED_ALG_LOWER 

    try:

        # Training DF default column names
        idx1_name = 'id1'
        idx2_name = 'id2'
        match_prob_unsup = 'Similarity_Unsup'
        match_prob_sup = 'Similarity_Sup'
        match_flag = 'is_match'

        training_df = pd.concat([feature_vector, probabilities], axis=1)
        training_df.rename_axis(index=[idx1_name, idx2_name], inplace=True)
        training_df.rename(columns={0: match_prob_unsup}, inplace=True)
        #training_df[match_prob_unsup] = training_df[match_prob_unsup].apply(lambda x: '%.17f' % x)
        training_df[match_prob_sup] = 'NA'
        training_df[match_flag] = 'NA'
        training_df.loc[training_df[match_prob_unsup] >= SUPERVIZED_ALG_UPPER, match_flag] = 1
        training_df.loc[training_df[match_prob_unsup] <= SUPERVIZED_ALG_LOWER, match_flag] = 0

        return {'status': True, 'data': training_df}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}', 'data': None}



def save_results_to_file(filePath, fileName, fileType, data_obj, indexFlag):

    
    try:

        if fileType == 'pkl':
            data_obj.to_pickle(f"{filePath}/{fileName}.pkl")

        elif fileType == 'csv':
            data_obj.to_csv(f"{filePath}/{fileName}.csv", encoding='utf-8-sig', float_format='%.9f', index=indexFlag)

        else:
            return {'status': False, 'err_msg': f"[INFO: {fileType} format is not supported]"}

    except Exception as e:
        return {'status': False, 'err_msg': '[ERROR: ' + str(e) + ']'}

    return {'status': True, 'msg': f"[INFO: File Saved]"}




def clustering_pairs(pairs_list):
    groupList = []
    tempList = pairs_list

    try:
        while len(tempList) > 1:

            ###### Get the first pair in the list
            group = [tempList[0]]
            srcPair = set(tempList[0])
            tempList.pop(0)

            ###### Loop and identify all the connected pairs in the list
            while (1 == 1):
                flag = 0
                indexList = []

                for ind in range(0, len(tempList)):

                    tgtPair = set(tempList[ind])
                    if len(set.intersection(srcPair, tgtPair)) > 0:
                        srcPair = srcPair.union(tgtPair)
                        group.append(tempList[ind])
                        indexList.append(ind)
                        flag = 1

                tempList = [i for j, i in enumerate(tempList) if
                            j not in indexList]

                if flag == 0:
                    break

            groupList.append(list(set(chain(*group))))

        if len(tempList) == 1:
            groupList.append(tempList[0])

        return {'status': True, 'data': groupList}

    except Exception as e:
        print(e)
        return {'status': False, 'err_msg': f'[Error: {str(e)}', 'data': None}


def generate_clustering_result(predictions, original_df):

    try:

        rst = clustering_pairs(predictions)


        if not rst['status']:
            return {'status': False, 'err_msg': rst['err_msg']}

        clusters = rst['data']

        # Get Records
        clustered_records = list(chain(*clusters))

        all_records = original_df.index.values.tolist()
        isolated_records = list(set(all_records).difference(set(clustered_records)))

        # Generate new DataFrame
        column_names =  list(original_df.columns.values) + ['ClusterID', 'RowID', 'TypeClass']

        # Initialize Clustered Data DF
        new_index_cl = range(0, len(clustered_records))
        df_clustered_data = pd.DataFrame(columns=column_names, index=new_index_cl)
        original_df['ClusterID'] = np.nan; original_df['RowID'] = np.nan; original_df['TypeClass'] = np.nan

        # Setting data in the new DF for matched records
        counter = 0

        for i in range(0, len(clusters)):
            for j in range(len(clusters[i])):
                df_clustered_data.iloc[counter] = original_df.iloc[clusters[i][j]]
                df_clustered_data['ClusterID'][counter] = i + 1
                df_clustered_data['TypeClass'][counter] = 'C'
                df_clustered_data['RowID'][counter] = clusters[i][j]
                counter += 1
        
         
        # Initialize Isolated Data DF
        new_index_iso = range(0, len(isolated_records))
        df_isolated_data = pd.DataFrame(columns=column_names, index=new_index_iso)

        # Setting data in the new DF for isolated records
        for i in range(0, len(isolated_records)):
            df_isolated_data.iloc[i] = original_df.iloc[isolated_records[i]]
            df_isolated_data['ClusterID'][i] = len(clusters) + i + 1
            df_isolated_data['TypeClass'][i] = 'I'
            df_isolated_data['RowID'][i] = isolated_records[i]


        # Merging data for both matched and isolated records, creating final DF
        df_dedup_data = pd.concat([df_clustered_data, df_isolated_data], axis=0, ignore_index=True)

        clus_data = {'Cluster DF': df_dedup_data, 'Cluster Info': {"total clusters": len(clusters),
                "clustered records": len(clustered_records), "isolated records": len(isolated_records)}}

        return {'status': True, 'data': clus_data}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: {str(e)}', 'data': None}




def pair_clustering_unsupervised(content):

    try:

        input_parameters = content['input_parameters']
        output_parameters = content['output_parameters']

        match_scores = None
        match_pairs = None
        match_eval = None

        ### Read Input File
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Reading the Input File]')
        
        #file_type = input_parameters['fileType'] if 'fileType' in input_parameters else 'pkl'
        
        #df_rst = get_input_data( input_parameters['filePath'], input_parameters['fileName'], file_type)
         
        #if not df_rst['status']:
        #    return {'status': False, 'err_msg': df_rst['err_msg']}

        df = input_parameters['data']
        col_types =  df.dtypes.to_dict()

        ### Indexing
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Setting Index]')
        indexer = recordlinkage.Index()
        
        if 'block_attributes' in content:
            block_fields = content['block_attributes']
            for field in block_fields:
                indexer.add(Block(field))
        else:
            indexer.add(Full())

        ### Generate Candidate Links
        candidate_links = indexer.index(df)

        ### Set Match Attributes
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Setting Match Attributes]')
        match_rst = set_match_attributes(content['match_attributes'], col_types)
        if not match_rst['status']:
            return {'status': False,
                    'err_msg': match_rst['err_msg']}
        match_fields = match_rst['data']

        ### Generate Comparison Vector
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Generating Comparison Vector]')
        cmp_rst = set_comparison_parameters(match_fields, candidate_links, df)

        if not cmp_rst['status']:
            return {'status': False,
                    'err_msg': f"[Error: set_comparison_parameters() failed with {cmp_rst['err_msg']}"}

        comparison_vector = cmp_rst['data']

        ### Determine the execution type and update feature vector, match scores and match pairs
        feature_vector = comparison_vector.copy()
        exec_type = content['execution_type'] if 'execution_type' in content else 'ecm'

        if exec_type == "weighted_score":

            print('[INFO :CLUSTERING UNSUPERVISED INFO: select weighted based algortihm,  calculating Weighted Mean and Matches]')
            threshold = content['similarity_threshold']
            ws_rst = weighted_score_process(feature_vector, match_fields, threshold)

            if not ws_rst['status']:
                return {'status': False,
                        'err_msg': f"[Error: weighted_score_process() failed with {ws_rst['err_msg']}"}

            match_scores = ws_rst['data']['similarity_scores']
            match_pairs = ws_rst['data']['match_pairs']

        else:
            model_parameters = {'binary_threshold': content['similarity_threshold']}
            
            if 'model_parameters' in content:
                model_parameters.update(content['model_parameters'][0])

            print('[INFO :CLUSTERING UNSUPERVISED INFO: select ECM algortihm, starting Classification]')
            cl_rst = recordlinkage_classifier_process(feature_vector, model_parameters)

            if not cl_rst['status']:
                return {'status': False, 'err_msg': f"[Error: recordlinkage_classifier_process() failed with {cl_rst['err_msg']}]"}

            match_scores = cl_rst['data']['match_probability']
            #match_pairs = cl_rst['data']['prediction']
            match_eval = cl_rst['data']['evaluation']


        print('[INFO :CLUSTERING UNSUPERVISED INFO: Generating Training Dataset]')
        train_rst = generate_training_set(comparison_vector, match_scores)

        if not train_rst['status']:
            return {'status': False,
                    'err_msg': f"[Error: generate_training_set() failed with {train_rst['err_msg']}]"}

        training_df = train_rst['data']

        # Save Training DF
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Saving Training Dataset]')
        training_data_param = output_parameters['training']

        training_file_path = training_data_param['filePath']
        training_file_type = training_data_param['fileType'] if 'fileType' in training_data_param else 'csv'
        training_file_name = training_data_param['fileName'] if 'fileName' in training_data_param else 'Training'
        
        save_results_to_file(training_file_path, training_file_name, training_file_type,
                             training_df, indexFlag=True)

        # Generate Clustering Result
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Generating Clustered Results]')
        training_df.reset_index(inplace=True)

        df_match = training_df.loc[training_df['Similarity_Unsup'] >= content['similarity_threshold'], ['id1', 'id2']]
        match_pairs = list(zip(list(df_match['id1']), list(df_match['id2'])))

        cluster_rst = generate_clustering_result(match_pairs, df)

        if not cluster_rst['status']:
            return {'status': False,
                    'err_msg': f"[Error: generate_clustering_result() failed with {cluster_rst['err_msg']}]"}



        clusters_df = cluster_rst['data']['Cluster DF']

        # Save Clustering DF
        print('[INFO :CLUSTERING UNSUPERVISED INFO: Saving Clustering Result]')

        prediction_data_param = output_parameters['prediction']

        prediction_file_path = prediction_data_param['filePath']
        prediction_file_type = prediction_data_param['fileType'] if 'fileType' in prediction_data_param else 'csv'
        prediction_file_name = prediction_data_param['fileName'] if 'fileName' in prediction_data_param else 'ClusteringResult'

        save_results_to_file(f'{prediction_file_path}', prediction_file_name, prediction_file_type,
                             clusters_df, indexFlag = False)
        #save_results_to_file(f'{prediction_file_path}', prediction_file_name,
        #                     prediction_file_type,
        #                     clusters_df, indexFlag=False)

        summary = {
            "num_row": clusters_df.shape[0],
            "num_duplicate_records": cluster_rst['data']['Cluster Info']["clustered records"],
            "num_isolate_records": cluster_rst['data']['Cluster Info']["isolated records"],
            "num_duplicate_clusters": cluster_rst['data']['Cluster Info']["total clusters"]
        }

        if match_eval:
            summary["model evaluation"] = match_eval['Model Evaluation']
        
        summary["similarity_threshold"] = content['similarity_threshold']
        summary["block_attributes"] = content['block_attributes'],
        summary["match_attributes"] = content['match_attributes']

        print('[INFO :CLUSTERING UNSUPERVISED INFO: Saving Metadata]')
         
        file = open(str(prediction_file_path) + "/summary.json", 'w')
        json.dump(summary, file)
        file.close()


        return {'status': True, 'data': summary, 'msg': f"[INFO: Clustering Process Complete]"}

    except Exception as e:
        return {'status': False, 'err_msg': f'[Error: pair_clustering_unsupervised() failed with {str(e)}]'}



def create_folder():
    path = f"./output/"
    if not os.path.exists(path):
        os.makedirs(path)


def data_summary(path):
    
    files = os.listdir(path)
    #print(files)
    files_list = []
    cols = ['file_name','VendorName','VendorName_conf',	'CustomerName',	'CustomerName_conf', 'VendorAddress', 'CustomerAddress', "Bing_Search_Description", "Bing_Search_KeyPhase", "Bing_Entity_Type", "Bing_Entity_Description", "Bing_Entity_KeyPhase", "Data_Category", "Data_Summarization", 'Description',	'Description_conf',	'Quantity',	'Quantity_conf','Unit',	'Unit_conf','Amount','Amount_conf']
    
    for fileName in files:
        
        print(f"{path}{fileName}")
        with open(f"{path}{fileName}", 'rb') as f2: 
            json_results = json.load(f2)
            
        for idx, invoice in enumerate(json_results['documents']): 

            temp = [json_results['file_name'], invoice.get("VendorName"), invoice.get("VendorName_conf"), invoice.get("CustomerName"), invoice.get("CustomerName_conf")]
            

            if invoice.get("VendorAddress"):
                try:
                    if invoice.get("VendorAddress").get("street_address"):
                        Vendor_address = f"{invoice['VendorAddress']['street_address']},{invoice['VendorAddress']['city']},{invoice['VendorAddress']['state']},{invoice['VendorAddress']['postal_code']}"
                    elif invoice.get("VendorAddress").get("po_box"):
                        Vendor_address = f"{invoice['VendorAddress']['po_box']},{invoice['VendorAddress']['city']},{invoice['VendorAddress']['state']},{invoice['VendorAddress']['postal_code']}"
                    else:
                        Vendor_address = None
                except:
                    Vendor_address = None
            else:
                Vendor_address = None
            temp.append(Vendor_address)
             

            if invoice.get("CustomerAddress"):
                try:
                    if invoice.get("CustomerAddress").get("street_address"):
                        Customer_address = f"{invoice['CustomerAddress']['street_address']},{invoice['CustomerAddress']['city']},{invoice['CustomerAddress']['state']},{invoice['CustomerAddress']['postal_code']}"
                    elif invoice.get("CustomerAddress").get("po_box"):
                        Customer_address = f"{invoice['CustomerAddress']['po_box']},{invoice['CustomerAddress']['city']},{invoice['CustomerAddress']['state']},{invoice['CustomerAddress']['postal_code']}"
                    else:
                        Customer_address = None
                except:
                    Customer_address = None
            else:
                Customer_address = None
            temp.append(Customer_address)

            temp.append(invoice.get("Bing_Search_Description"))
            temp.append(invoice.get("Bing_Search_KeyPhase"))
            temp.append(invoice.get("Bing_Entity_Type"))
            temp.append(invoice.get("Bing_Entity_Description"))
            temp.append(invoice.get("Bing_Entity_KeyPhase"))
            temp.append(invoice.get("Data_Category"))
            temp.append(invoice.get("Data_Summarization"))
        

            if invoice.get("Items"):
                for idx2, Singleitem in enumerate(invoice.get("Items")): 
                    print(idx2)
                
                    item_list = copy.copy(temp)
                    item_list.append(Singleitem.get('Description'))
                    item_list.append(Singleitem.get('Description_conf'))
                    item_list.append(Singleitem.get('Quantity'))
                    item_list.append(Singleitem.get('Quantity_conf'))
                    item_list.append(Singleitem.get('UnitPrice'))
                    item_list.append(Singleitem.get('UnitPrice_conf'))
                    item_list.append(Singleitem.get('Amount'))
                    item_list.append(Singleitem.get('Amount_conf'))
                    print(item_list)
                    files_list.append(item_list)

             
        print(files_list)
    
    df = pd.DataFrame(files_list, columns =cols)
    df['file_type'] = 1
    print(df.head())
    return df




def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--similarity_threshold", type=float, default=0.8)
    parser.add_argument('--block_attributes',   type=str, nargs='+')
    parser.add_argument('--cluster_attributes',  type=str, nargs='+')
    parser.add_argument('--input', type=str)
    #parser.add_argument('--file_name', type=str)
    #parser.add_argument('--file_type', type=str)
    parser.add_argument('--output', type=str)



    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):

    global clusteirng_algorithm

    content = {}
    content['execution_type'] = clusteirng_algorithm  ### modes:  weighted_score or ecm
    content['similarity_threshold'] = args.similarity_threshold
    content["block_attributes"]  = args.block_attributes
    content["match_attributes"] =  args.cluster_attributes

    
    path = f"{Path(args.input)}/categorized_json/"
    df = data_summary(path)


    content['input_parameters'] = {
            #"filePath": args.file_path_in,
            #"fileName": args.file_name,
            #"fileType": args.file_type,
            "data": df 
    }

    content['output_parameters'] = {
            "training": 
            {
            "filePath":  Path(args.output),
            "fileName": f'training',
            "fileType": 'csv'
            },
            "prediction": 
            {
            "filePath": Path(args.output),
            "fileName": f'clustering',
            "fileType": 'csv'
            }            

    }

     
    
    print(content)
    start = timer()
    #create_folder(args.file_path_in)
    create_folder()

    dt = pair_clustering_unsupervised(content)
    end = timer()
    #print("Total Time:", end-start)
    response = {
            'status': 'failure',
            'err_msg': ''
        }

    if dt['status']:
        response['result'] = {}
        response['result']['metadata'] = dt['data']
        response['result']['metadata']['Time Elapsed'] = str(end-start)
        response['status'] = 'success'
        print(response)
        return response
    else:
        response['err_msg'] = dt['err_msg']
        print(response)
        return response


if __name__ == "__main__":
    # parse args
    args = parse_args()
    ## run main function
    main(args)
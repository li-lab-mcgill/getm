import numpy as np
import pandas as pd
from collections import defaultdict



def create_dict(data, attribute_name, tag):
    '''
    Create Boolean dictionary for certain attributes
    :param data: data frame of individuals with columns: <user_id, attribute_id>
    :param attribute_name: name of attribute
    :param tag: specific attribute
    :return: dictionary with key to be subject of interest and value to be Boolean for tag
    '''
    dict_ = {}
    for _, row in data.iterrows():
        user = row["user_id"]
        attr = row[attribute_name]
        if attr == tag:
            dict_[user] = 1
        else:
            dict_[user] = 0
    return dict_
def create_filtered_set(original_list, related_list):
    '''
    Create filtered medication/condition set
    :param original_list: a list that contains all unique medications/conditions
    :param related_list: a list that contains medications/conditions relevant to pain of interest
    :return: a filtered list of which the relevant medications/conditions have been removed
    '''
    if not related_list:
        return original_list
    else:
        return list(set(original_list).difference(set(related_list)))

def combine_data(med_data, cond_data):
    '''
    combine individuals' medication data and condition data
    :param med_data: data frame of individuals' medication info with columns:<user_id, med_ids>
    :param cond_data: data frame of individuals' condition info with columns:<user_id, cond_ids>
    :return data: combined dataframe
    '''
    med_user = set(med_data["user_id"])
    cond_user = set(cond_data["user_id"])
    med_id_list = []
    cond_id_list = []
    user_ids = []

    for _, row in med_data.iterrows():
        user_id = row["user_id"]
        user_ids.append(user_id)
        med_id_list.append(row["med_ids"])
        if user_id in cond_user:
            idx = cond_data.loc[cond_data['user_id'] == user_id].index[0]
            cond_ids = cond_data.iloc[idx]["cond_ids"]
            cond_id_list.append(cond_ids)
        else:
            cond_id_list.append([])
    cond_only_user = cond_user.difference(med_user)
    for user in cond_only_user:
        user_ids.append(user)
        idx = cond_data.loc[cond_data['user_id'] == user].index[0]
        cond_ids = cond_data.iloc[idx]["cond_ids"]
        cond_id_list.append(cond_ids)
        med_id_list.append([])
    df = pd.DataFrame(data={"user_id": user_ids, "cond_ids": cond_id_list, "med_ids": med_id_list})
    return df



def create_input_file(data, med_list, cond_list):
    '''
    Create input file for GETM
    :param data: data frame of individuals' medication and condition info with columns:<user_id, med_ids, cond_ids>
    :param med_list: the medication list of length M
    :param cond_list: the condition list of length C
    :return arr: input data in dimension N*(M+C)
    :return med_idx: medication index dictionary in arr
    :return cond_idx: condition index dictionary in arr
    '''
    med_idx = {k: v for v, k in enumerate(med_list)}
    cond_idx = {k: v+len(med_idx) for v, k in enumerate(cond_list)}
    arr = np.zeros((len(data), len(med_list)+len(cond_list)))
    for idx, row in data.iterrows():
        meds = row["med_ids"]
        conds = row["cond_ids"]
        for med in meds:
            if med in med_idx:
                arr[idx][med_idx[med]] += 1
        for cond in conds:
            if cond in cond_idx:
                arr[idx][cond_idx[cond]] += 1
    return arr, med_idx, cond_idx

def create_cond_graph(data, saved_file):
    '''
    Create condition knowledge graph
    :param data: data frame of condition hierarchical information with columns <node_id, parent_id>
    :param saved_file: data path to save output
    :return: condition graph data in the format of <node1, node2>
    '''
    pairs = set([])
    for _, row in data.iterrows():
        node_id = row["node_id"]
        parent_id = row["parent_id"]
        if parent_id != None:
            if (parent_id, node_id) not in pairs:
                pairs.add((node_id, parent_id))
    pairs = list(pairs)
    df = pd.DataFrame(data={"node1": pairs[:, 0], "node2": pairs[:, 1]})
    df.to_csv(saved_file, index=None, sep=" ")
    return df

def create_med_graph(atc_data, uk_med_dict, saved_file):
    '''
    Create medication knwoledge graph
    :param atc_data: ATC classification system
    :param uk_med_dict: UK Biobank {medication: code} dictionary
    :param saved_file: data path to save output
    :return: medication graph data in the format of <node1, node2>
    '''

    pairs = set([])
    for _, row in atc_data:
        if len(row["code"]) == 8:
            atc_code = row["code"]
            name = row["name"]
            if name in uk_med_dict:
                code = uk_med_dict[name]
            else:
                code = atc_code
            pairs.add((code, atc_code[:5]))
            pairs.add((atc_code[:5], atc_code[:4]))
            pairs.add((atc_code[:4], atc_code[:3]))
            pairs.add((atc_code[:3], atc_code[:1]))

    df = pd.DataFrame(data={"node1": pairs[:, 0], "node2": pairs[:, 1]})
    df.to_csv(saved_file, index=None, sep = " ")
    return df









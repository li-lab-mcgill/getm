import numpy as np
import pandas as pd
import pickle



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
    all_user = med_user.union(cond_user)
    data = [[[], []] for _ in range(len(all_user))]
    user_index_dict = {u: i for i, u in enumerate(list(all_user))}
    for _, row in med_data.iterrows():
        uid = user_index_dict[row["user_id"]]
        med = row["med_ids"]
        data[uid][0].append(med)
    for _, row in cond_data.iterrows():
        uid = user_index_dict[row["user_id"]]
        cond = row["cond_ids"]
        data[uid][1].append(cond)
    data = np.array(data, dtype=object)
    df = pd.DataFrame(data={"user_id": list(all_user), "med_ids": data[:, 0], "cond_ids": data[:, 1]})
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
    arr = np.zeros((data['user_id'].nunique(), len(med_list)+len(cond_list)))
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
    node_index_dict = {}
    idx = 0
    pairs = set([])
    for _, row in data.iterrows():
        node_id = row["node_id"]
        parent_id = row["parent_id"]
        if node_id not in node_index_dict:
            node_index_dict[node_id] = idx
            idx += 1
        if parent_id not in node_index_dict:
            node_index_dict[parent_id] = idx
            idx += 1
        if parent_id != None:
            if (parent_id, node_id) not in pairs:
                pairs.add((node_index_dict[node_id], node_index_dict[parent_id]))
    pairs = list(pairs)
    pairs=np.array(pairs)
    df = pd.DataFrame(data={"node1": pairs[:, 0], "node2": pairs[:, 1]})
    df.to_csv(f"{saved_file}/cond_graph.txt", index=None, header=None, sep=" ")
    with open(f"{saved_file}/cond_index_dict.pickle", "wb") as f:
        pickle.dump(node_index_dict, f)
    return df


def create_med_graph(atc_data, uk_med_dict, saved_file):
    '''
    Create medication knwoledge graph
    :param atc_data: ATC classification system
    :param uk_med_dict: UK Biobank {medication: code} dictionary
    :param saved_file: data path to save output
    :return: medication graph data in the format of <node1, node2>
    '''
    node_index_dict = {}
    idx = 0
    pairs = set([])
    for _, row in atc_data.iterrows():
        if len(row["code"]) == 7:
            atc_code = row["code"]
            name = row["name"]
            if atc_code in uk_med_dict:
                code = uk_med_dict[atc_code]
            else:
                code = atc_code

            if code not in node_index_dict:
                node_index_dict[code] = idx
                idx += 1
            if atc_code[:5] not in node_index_dict:
                node_index_dict[atc_code[:5]] = idx
                idx += 1
            if atc_code[:4] not in node_index_dict:
                node_index_dict[atc_code[:4]] = idx
                idx += 1
            if atc_code[:3] not in node_index_dict:
                node_index_dict[atc_code[:3]] = idx
                idx += 1
            if atc_code[:1] not in node_index_dict:
                node_index_dict[atc_code[:1]] = idx
                idx += 1

            pairs.add((node_index_dict[code], node_index_dict[atc_code[:5]]))
            pairs.add((node_index_dict[atc_code[:5]], node_index_dict[atc_code[:4]]))
            pairs.add((node_index_dict[atc_code[:4]], node_index_dict[atc_code[:3]]))
            pairs.add((node_index_dict[atc_code[:3]], node_index_dict[atc_code[:1]]))
    pairs = np.array(list(pairs))
    df = pd.DataFrame(data={"node1": pairs[:, 0], "node2": pairs[:, 1]})
    df.to_csv(f"{saved_file}/med_graph.txt", index=None, header=None, sep=" ")
    with open(f"{saved_file}/med_index_dict.pickle", "wb") as f:
        pickle.dump(node_index_dict, f)
    return df









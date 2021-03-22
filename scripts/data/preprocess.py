import os
import numpy as np 
import csv
import pickle as pkl
import scipy.io
from scipy.sparse import csc_matrix
from tqdm import tqdm
import pickle
from IPython import embed

data_dir = '../../../dynamicMixEHR/data/'
class DIN:
    def __init__(self):
        # din reference
        din_list = []
        with open(data_dir+"drug_product.csv") as f:
            din_reader = csv.reader(f)
            for row in din_reader:
                din_list.append(row)
        # code, cate, class, din, brand, descriptor, pediatric_flag, access_No, .....
        din2drug_list = np.array(din_list[1:])
        din2drug = dict(zip(din2drug_list[:,3], din2drug_list[:,0]))
        drug2ingredient_list = []
        with open(data_dir+"active_ingredients.csv") as f:
            drug_reader = csv.reader(f)
            for row in drug_reader:
                drug2ingredient_list.append(row)
        # drug code, active_ing_code, ingredient, ingredient_supplied_ind, strength, .....
        drug2ingredient_list = np.array(drug2ingredient_list[1:])
        drug2ingredient = {}
        for row in drug2ingredient_list:
            if not row[0] in drug2ingredient:
                drug2ingredient[row[0]] = row[2]
            else:  # one drug may includes more than one ingredients, refering to multiple cols
                drug2ingredient[row[0]] += '/' + row[2]
        self.din2drug = din2drug
        self.drug2ingredient = drug2ingredient

    def query(self, din):
        try:
            return self.drug2ingredient[self.din2drug[din]]
        except:
            return 'Not Found'
din_dict = DIN()
SMALL = None # 10000000

def is_COPD_icd(icd_token, token2icd):
    icd_9_code = token2icd[icd_token]
    return icd_9_code[:3] == '491' or icd_9_code[:3]=='492' or icd_9_code[:3]=='496'

def count_occurence(mode, path, save_path):
    print(mode +'   ' + path)

    occurence = {}
    patient = set()
    with open(path) as f:
        f_reader = csv.reader(f)
        for i, row in tqdm(enumerate(f_reader)):
            if SMALL and i >= SMALL:
                break

            if i == 0:
                if mode == 'drug_ingredient':
                    code_col = dict(zip(row, list(range(len(row)))))['din']
                else:
                    code_col = dict(zip(row, list(range(len(row)))))[mode]
                continue

            code = row[code_col]
            if mode == 'drug_ingredient':   # more than one ingredient mapped by one din
                codes = din_dict.query(code).split('/')
                for code in codes:
                    if not code in occurence:
                        occurence[code] = set()
                    occurence[code].add(row[0])
            else:
                if not code in occurence:
                    occurence[code] = set()
                occurence[code].add(row[0])

            patient.add(row[0])
    print('# (EHR of {}): {}'.format(mode, i))
    print('# (patient): {}'.format(len(patient)))
    # id2patient = dict(zip(list(patient), list(range(len(patient)))))
    # with open(mode + '_' + "id2patient.pkl", 'wb') as f:
        # pickle.dump(id2patient, f)

    count = {}
    for code in tqdm(occurence.keys()):
        count[code] = len(occurence[code])
    if '' in count:
        count.pop('')
    print('# ({} code): {}'.format(mode, len(count)))
    print('sum of occurences: {}'.format(sum(count.values())))

    cnt = np.array(list(count.values()))
    idx = np.argsort(cnt)
    cnt = cnt[idx]
    code = np.array(list(count.keys()))[idx]
    pickle.dump(np.vstack((code, cnt/len(patient))), open(save_path+mode+'_frequency.pkl', 'wb'))


def filter_ehr(save_path, data_file, mode, freq_path, threshold): 
    # save_path = 'small/'
    np.random.seed(42)
    if SMALL:
        threshold = 1

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    freq = pickle.load(open(freq_path + mode + '_frequency.pkl', 'rb'))
    print(save_path)
    print(mode + ' threshold: ' + str(threshold))

    # non-COPD case: 
    chosen_cnt = np.nonzero(freq[1].astype('float')>threshold)[0][0]
    
    # COPD case:
    # if mode == 'icd':
    #     chosen_cnt = freq.shape[1] - 3
    #     print('dropped: '+str(list(freq[0,-3:])))
    # else:
    #     chosen_cnt = freq.shape[1]
    code_set = set(freq[0,:chosen_cnt])
    if mode == 'drug_ingredient':
        code_set -= set('Not Found')
    print('#(total):  {}      #(chosen (vocab)): {}'.format(freq.shape[1], chosen_cnt))

    code_dict = dict(zip(code_set, np.arange(chosen_cnt)))
    with open(save_path+mode+'code_dict.pkl', 'wb') as f:
        pkl.dump(code_dict, f)
    # token2code = dict(zip(np.arange(len(code_set)), code_set))
    pkl.dump(list(code_set), open(save_path+mode+'_vocab.pkl', 'wb'))

    # map: code to token, date to year
    EHR = []
    with open(data_file) as f:
        f_reader = csv.reader(f)
        for i, row in tqdm(enumerate(f_reader)):
            if SMALL and i >= SMALL:
                break

            if i == 0:
                name_col = dict(zip(row, list(range(len(row)))))
                if mode == 'drug_ingredient':
                    code_col = name_col['din']
                else:
                    code_col = name_col[mode]
                date_col = name_col['date']
                id_col = name_col['id']
                continue
            else:
                code = row[code_col]
                if mode == 'drug_ingredient':
                    codes = din_dict.query(code).split('/')
                    for code in codes:
                        if code in code_set:
                            EHR.append([int(row[id_col]), code_dict[code], int(row[date_col][:4]), mode])
                else:
                    if code in code_set:
                        EHR.append([int(row[id_col]), code_dict[code], int(row[date_col][:4]), mode])


    print('#(filtered EHR): ' + str(len(EHR)))
    print(EHR[0]) # patient code time

    # EHR = np.array(EHR, dtype=object)  

    return EHR
    
def is_COPD_patient(EHR, token2code_icd):
    # EHR:  0 - patient, 1 - code, 2 - time (year), 3 - code type
    for ehr in EHR:
        if ehr[3] == 'icd' and is_COPD_icd(ehr[1], token2code_icd):
            return True
    return False

def split_save_file(save_path, EHR):
    # save_path = 'small/'
    # map & sort: patient (source) & time
    # EHR:  0 - patient, 1 - code, 2 - time (year), 3 - code type
    EHR = np.array(EHR, dtype=object)
    EHR = EHR[np.argsort(EHR[:,0])]  # sorted by patient
    patients, p_idx, p_cnt = np.unique(EHR[:,0], return_index=True, return_counts=True)
    print('#(filtered patients): {}'.format(len(patients)))
    patient_id_1stad = []
    EHR_combined_by_years = []
    tokens_combined_by_years = []
    counts_combined_by_years = []
    source_time_slot = []
    # for COPD patient (icd ^(491|492|496)) filtering  
    # with open(save_path+'code_dict.pkl', 'wb') as f:
    #     pkl.dump(code_dict, f)
    # token2code = dict(zip(np.arange(len(code_set)), code_set))
    # with open(save_path+'icd_code_dict.pkl', 'rb') as f:
    #     code_dict_icd = pkl.load(f)
    # token2code_icd = dict(zip(list(code_dict_icd.values()), list(code_dict_icd.keys())))
    # COPD_patient = 0
    for pi, patient in tqdm(enumerate(patients)):

        patient_records = EHR[p_idx[pi]:p_idx[pi]+p_cnt[pi]]

        patient_records = patient_records[np.argsort(patient_records[:,2])] # sorted by time
        # if not is_COPD_patient(patient_records, token2code_icd):
            # continue
        # COPD_patient+=1
    
        time_begin = patient_records[0,2]
        patient_records[:,2] -= time_begin
        time_slot, t_idx, t_cnt = np.unique(patient_records[:,2], return_index=True, return_counts=True)
        patient_id_1stad.append([patient, time_begin, time_slot[-1]])
        # patient_records[:,0] = pi

        for ti, time in enumerate(time_slot):
            EHR_combined_by_years.append(patient_records[t_idx[ti], [0,2,3] ])  # patient-time-type
            docs = dict(zip(['icd', 'act_code', 'drug_ingredient'], [[] for _ in range(3)]))
            tokens, counts = {}, {}

            for ehr in patient_records[t_idx[ti]:t_idx[ti]+t_cnt[ti]]:
                docs[ehr[3]].append(ehr[1])
            for code in ['icd', 'act_code', 'drug_ingredient']:
                if len(docs[code]) > 0:
                    tokens[code], counts[code] = np.unique(docs[code], return_counts=True)
            tokens_combined_by_years.append(tokens)
            counts_combined_by_years.append(counts)

        num_time_slot = len(time_slot) 
        source_time_slot.append(num_time_slot)

    # print('#(COPD_patient): ' + str(COPD_patient))
    with open(save_path+'patient_id_admission.pkl', 'wb') as f:
        pickle.dump(patient_id_1stad, f)
    EHR_combined_by_years = np.array(EHR_combined_by_years)
    tokens_combined_by_years = np.array(tokens_combined_by_years)
    counts_combined_by_years = np.array(counts_combined_by_years)
    print('max time span: {}'.format(max(source_time_slot)))
    print('# (documents): {}'.format(sum(source_time_slot)))
    # return None


    # split to (train, valid, test)
    shuffled_patients = np.random.permutation(patients)
    train_patients = shuffled_patients[:int(0.6 * len(patients))]
    valid_patients = shuffled_patients[int(0.6 * len(patients)):int(0.9 * len(patients))]
    test_patients = shuffled_patients[int(0.9 * len(patients)):]

    split_set = dict(zip(list(train_patients) + list(valid_patients) + list(test_patients),
        ['train']*len(train_patients) + ['valid']*len(valid_patients) + ['test']*len(test_patients)))

    tokens = {'train': [], 'valid': [], 'test': []}
    counts = {'train': [], 'valid': [], 'test': []}
    timestamps = {'train': [], 'valid': [], 'test': []}
    sources = {'train': [], 'valid': [], 'test': []}
    types = {'train': [], 'valid': [], 'test': []}
    # labels = {'train': [], 'valid': [], 'test': []}
    for i, ehr in tqdm(enumerate(EHR_combined_by_years)):  # p, time, type
        split = split_set[ehr[0]]
        tokens[split].append(tokens_combined_by_years[i])
        counts[split].append(counts_combined_by_years[i])
        timestamps[split].append(ehr[1])
        sources[split].append(ehr[0])
        types[split].append(ehr[2])

    print('train: {}, valid: {}, test: {}'.format(len(train_patients), len(valid_patients), len(test_patients)))

    for split, abbr in zip(['train', 'valid', 'test'], ['tr','va','ts']):
        pickle.dump(tokens[split], open(save_path+'bow_'+abbr+'_tokens.pkl', 'wb'))
        pickle.dump(counts[split], open(save_path+'bow_'+abbr+'_counts.pkl', 'wb'))
        pickle.dump(timestamps[split], open(save_path+'bow_'+abbr+'_timestamps.pkl', 'wb'))
        pickle.dump(sources[split], open(save_path+'bow_'+abbr+'_sources.pkl', 'wb'))
        pickle.dump(types[split], open(save_path+'bow_'+abbr+'_types.pkl', 'wb'))



def generate_label_patients_mortality(data_path, save_path):
    label_dict = {}
    with open('raw/patients.csv') as f:
        f_reader = csv.reader(f)
        for i, row in enumerate(f_reader):
            if i == 0:
                code_col = dict(zip(row, list(range(len(row)))))
                continue
            label_dict[int(row[code_col['id']])] = row[code_col['month_of_death']] != ''

    print('mortality label\n # (patients): {}\n # (death): {}'.format(len(label_dict), sum(list(label_dict.values()))))

    with open(save_path+'.pkl', 'wb') as f:
        pickle.dump(label_dict, f)

def generate_patients_age(data_path, save_path):
    birth_dict = {}
    with open(data_dir+'raw/patients.csv') as f:
        f_reader = csv.reader(f)
        for i, row in enumerate(f_reader):
            if i == 0:
                code_col = dict(zip(row, list(range(len(row)))))
                continue
            birth_dict[int(row[code_col['id']])] = int(row[code_col['month_of_birth']][:4])

    lastad_age_dict = {}
    firstad_age_dict = {}
    with open(os.path.join(data_path,'patient_id_admission.pkl'), 'rb') as f:
        patient_ad = pickle.load(f)
        for p in patient_ad:
            lastad_age_dict[p[0]] = p[2]+p[1] - birth_dict[p[0]]
            firstad_age_dict[p[0]] = p[1] - birth_dict[p[0]]
    
    age_list = np.array(list(lastad_age_dict.values()))
    print('total: '+str(len(age_list)))
    print('#(age in [0, 10)): '+str(sum(age_list<10)))
    print('#(age in [10, 20)): '+str( sum(age_list<20) - sum(age_list<10) ))
    print('#(age in [20, 30)): '+str( sum(age_list<30) - sum(age_list<20) ))
    print('#(age in [30, 40)): '+str( sum(age_list<40) - sum(age_list<30) ))
    print('#(age in [40, 50)): '+str( sum(age_list<50) - sum(age_list<40) ))
    print('#(age in [50, 60)): '+str( sum(age_list<60) - sum(age_list<50) ))
    print('#(age in [60, 70)): '+str( sum(age_list<70) - sum(age_list<60) ))
    print('#(age in [70, 80)): '+str( sum(age_list<80) - sum(age_list<70) ))
    print('#(age in [80, 90)): '+str( sum(age_list<90) - sum(age_list<80) ))
    print('#(age in [90, 100)): '+str( sum(age_list<100) - sum(age_list<90) ))
    print('#(age in [100, )): '+str( len(age_list) - sum(age_list<100) ))
    # with open(save_path + '.pkl', 'wb') as f:
    #     pickle.dump(lastad_age_dict, f)
    with open(save_path + 'age_firstad.pkl', 'wb') as f:
        pickle.dump(firstad_age_dict, f)
        

def generate_patients_agestamps(data_path, save_path):
    with open(data_path+'age_firstad.pkl', 'rb') as f:
        age_dict = pickle.load(f)
    print(max(list(age_dict.values())))
    for abbr in ['tr', 'va', 'ts']: 
        sources = np.array(pickle.load(open(os.path.join(data_path, 'bow_'+abbr+'_sources.pkl'), 'rb')))
        timestamps = np.array(pickle.load(open(os.path.join(data_path, 'bow_'+abbr+'_timestamps.pkl'), 'rb')))

        ages = []
        for p, t in tqdm(zip(sources, timestamps)):
            ages.append(t+age_dict[p])

        ages = (np.array(ages)).astype('int')

        print(max(ages))
        with open(os.path.join(save_path, (abbr+'_agestamps.pkl')), 'wb') as f:
            pickle.dump(ages, f)

            
# if __name__ == '__main__':
### Stage 1 
# for mode, file_name in zip(['icd', 'act_code', 'drug_ingredient'], ['services.csv', 'services.csv', 'drugs.csv']):
#     if SMALL:
#         count_occurence(mode, path=data_dir+'raw/'+file_name, save_path='raw/small/')
#     else:
#         count_occurence(mode, data_dir+'raw/'+file_name, 'raw/')
#         # count_occurence(mode, 'raw_COPD/'+file_name, 'raw_COPD/'+mode+'_frequency.pkl')
# print('\n')
    
# Stage 2
EHR = []
for mode, file_name, thr in zip(['icd', 'act_code', 'drug_ingredient'], ['services.csv', 'services.csv', 'drugs.csv'], [0.02, 0.02, 0.05]):
    if SMALL:
        EHR += filter_ehr(save_path='small/', data_file=data_dir+'raw/'+file_name, mode=mode, freq_path='raw/small/', threshold=thr)
    else:
        EHR += filter_ehr(save_path='out-patient/', data_file=data_dir+'raw/'+file_name, mode=mode, freq_path='raw/', threshold=thr)
        # EHR += filter_ehr(save_path='COPD_out-patient/', data_file='raw_COPD/'+file_name, code=mode, freq_path='raw_COPD/', threshold=thr)
if SMALL:
    pickle.dump(EHR, open('small/EHR.pkl', 'wb'))
else:
    pickle.dump(EHR, open('EHR.pkl', 'wb'))
print('\n')

# Stage 3:
if SMALL:
    split_save_file(save_path='small/', EHR=EHR)
else:
    # split_save_file(save_path='COPD_out-patient/', EHR=EHR)
    split_save_file(save_path='out-patient/', EHR=EHR)
    pass
print('\n')
    
    

# Additional: mortality label
# generate_label_patients_mortality(data_path='raw/patients.csv', save_path='mortality_nontemporal')

# Additional: age label
if SMALL:
    generate_patients_age(data_path='small/', save_path='small/')
else:
    generate_patients_age(data_path='out-patient/', save_path='out-patient/')
    # generate_patients_age(data_path='COPD_out-patient/', save_path='COPD_out-patient/')
print('\n')

# Stage 4: age stamp for each admission  (should be used along with generate_patients_age function)
if SMALL:
    generate_patients_agestamps(data_path='small/', save_path='small/')
else:
    generate_patients_agestamps(data_path='out-patient/', save_path='out-patient/')
    # generate_patients_agestamps(data_path='COPD_out-patient/', save_path='COPD_out-patient/')

            
            

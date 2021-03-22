import os, time
import numpy as np
import pickle
from tqdm import tqdm
from scipy.sparse import coo_matrix
import sys
from IPython import embed

code_types = ['icd', 'act_code', 'drug_ingredient']

def collect_by_timestamps(tokens, counts, sources, times):
    # Pre-processing
    times[times>=100] = 100  # age upper bound
    times = np.floor(times/10)  # FIXME:  age threshold 
    filtered = times >= 0
    tokens, counts, sources, times = tokens[filtered], counts[filtered], sources[filtered], times[filtered]


    s_and_t = sources * 100 + times    # encode source & time to one array
    idx = np.argsort(s_and_t)
    _st, st_idx, st_cnt = np.unique(s_and_t[idx], return_index=True, return_counts=True)
    _tokens = []
    _counts = []
    print(len(_st))
    for s, i, c in tqdm(zip(_st, st_idx, st_cnt)):
        _token, _count = {}, {}
        for code in code_types:
            # FIXME: consider counts
            _token[code] = dict()
            # c = int(c*earlier_proportion)
            for ii in idx[i:i + c]:
                if not code in tokens[ii]:
                    continue
                for token, count in zip(tokens[ii][code], counts[ii][code]):
                    if token in _token[code]:
                        _token[code][token] += count
                    else:
                        _token[code][token] = count
            _count[code] = np.array(list(_token[code].values()))
            _token[code] = np.array(list(_token[code].keys()))

            # not consider counts
            # _t_code = [tokens[ii][code] for ii in sources_idx[i:i+c] if code in tokens[ii]]
            # if len(_t_code):
            #     _t[code] = np.unique(np.hstack(_t_code))
            #     _c[code] = np.ones(len(_t[code]))
        _tokens.append(_token)
        _counts.append(_count)
    _sources, _times = np.divmod(_st, 100)
    return _tokens, _counts, _sources, _times

def to_matrix(tokens, counts, sources, times, T, vocab_size):
    idx = np.argsort(sources)
    tokens, counts, sources, times = np.array(tokens)[idx], np.array(counts)[idx], np.array(sources)[idx], np.array(times)[idx]
    p, p_idx, p_cnt = np.unique(sources, return_index=True, return_counts=True)

    vocab_cum = np.cumsum([0]+vocab_size)
    V = sum(vocab_size)
    
    data, data_t = [], []
    cumulate = coo_matrix(np.tril(np.ones((T,T))))
    for pi, i, c in tqdm(zip(p, p_idx, p_cnt)):
        d, row, col = [], [], [] 
        for ttoken, ccount, ttime in zip(tokens[i:i+c], counts[i:i+c], times[i:i+c]):
            for ci, code in enumerate(code_types):
                if code in ttoken:
                    col += list(ttoken[code]+vocab_cum[ci])
                    d += list(ccount[code])
                    row += [ttime] * len(ttoken[code])
        p_mat = coo_matrix((d, (row, col)), shape=(T, V))
        p_mat_t = coo_matrix(cumulate.__mul__(p_mat))
        data.append(p_mat)
        data_t.append(p_mat_t)
    # FIXME: maybe T x sparse(P x V)  is better
    
    data, data_t = np.array(data), np.array(data_t)
    return data, data_t
        
def _fetch(path, save_path, name, abbr, temporal, predict, vocab):
    if name == 'test':
        return {'tokens': None, 'counts': None, 'times': None, 'sources': None, 'labels': None}
    tokens = np.array(pickle.load(open(os.path.join(path, 'bow_'+abbr+'_tokens.pkl'), 'rb')))
    counts = np.array(pickle.load(open(os.path.join(path, 'bow_'+abbr+'_counts.pkl'), 'rb')))
    sources = np.array(pickle.load(open(os.path.join(path, 'bow_'+abbr+'_sources.pkl'), 'rb')))

    if temporal:
        times = np.array(pickle.load(open(os.path.join(path, abbr+'_agestamps.pkl'), 'rb')))
        tokens, counts, sources, times = collect_by_timestamps(tokens, counts, sources, times)
    else:
        # collect all years' visit together
        tokens, counts, sources = collect_all_years(tokens, counts, sources)
        if predict:
            tokens_1, counts_1, _ = collect_all_years(tokens, counts, sources, 1)
        times = np.zeros(len(sources))

    if predict:
        # mortality prediction for ETM 
        label_dict = pickle.load(open('data/mortality_nontemporal.pkl', 'rb'))
        labels = np.array([label_dict[s] for s in sources])
    else:
        labels = np.zeros(len(sources))

    # FIXME: didn't consider labels

    # transfer to P x T x concat(codes)
    vocab_size = [len(vocab[code]) for code in code_types]
    T = 11
    data, data_t = to_matrix(tokens, counts, sources, times, T, vocab_size)
    if name == 'valid':
        name = 'test'
    np.save(open(save_path+'bow_'+name+'.npy', 'wb'), data)
    np.save(open(save_path+'bow_t_'+name+'.npy', 'wb'), data_t)
    
    return data, data_t

def get_data(path, save_path, temporal, predict):
     ### load vocabulary multi_modal = args.multi_modal
    vocab = {}
    for code in code_types:
        vocab[code] = pickle.load(open(os.path.join(path, code+'_vocab.pkl'), 'rb'))

    vocab_size = [len(vocab[code]) for code in code_types]
    print(code_types)
    print(vocab_size)
    np.savetxt(os.path.join(save_path,'metadata.txt'), np.array([code_types, vocab_size, [1]*len(code_types), ['*']*len(code_types)], dtype=str), fmt='%s')

    train = _fetch(path, save_path,'train', 'tr', temporal, predict, vocab)
    valid = _fetch(path, save_path, 'valid', 'va', temporal, predict, vocab)
    # test = _fetch(path, save_path, 'test', 'ts', temporal, predict, vocab)

data_path = sys.argv[1]
save_path = sys.argv[2]
get_data(data_path, save_path, True, False)
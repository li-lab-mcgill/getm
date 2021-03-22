import os
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import csv
import xlsxwriter
from pandas import ExcelWriter
from wordcloud import WordCloud, STOPWORDS

from IPython import embed

# from embed.queryICD import queryICD, ICD2disease

# ICD reference
ICD_list = []
with open('data/icd9_qc.csv') as f:
    icd_reader = csv.reader(f)
    for row in icd_reader:
        ICD_list.append(row)
 # ICD Disc_en Disc_fr
ICD_list = np.array(ICD_list[1:])
ICD_dict = dict(zip(ICD_list[:,0], list(zip(ICD_list[:,1], ICD_list[:,2]))))
def queryICD(icd):
    if icd in ICD_dict:
        if ICD_dict[icd][0] != '':
            return ICD_dict[icd][0][:50]
        else:
            return ICD_dict[icd][1][:50]
    else:
        return 'Not Found'


# acte reference
act_list = []
with open('data/actes.csv') as f:
    act_reader = csv.reader(f)
    for row in act_reader:
        act_list.append(row)
# spec, code, nom, description
act_list = np.array(act_list[1:])
act_dict = dict(zip(act_list[:,1], act_list[:,2]))
def queryACT(act):
    if act in act_dict:
        return act_dict[act]
    else:
        return 'Not Found'


# din reference
din_list = []
with open("data/drug_product.csv") as f:
    din_reader = csv.reader(f)
    for row in din_reader:
        din_list.append(row)
# code, cate, class, din, brand, descriptor, pediatric_flag, access_No, .....
din2drug_list = np.array(din_list[1:])
din2drug = dict(zip(din2drug_list[:,3], din2drug_list[:,0]))

drug2ingredient_list = []
with open("data/active_ingredients.csv") as f:
    drug_reader = csv.reader(f)
    for row in drug_reader:
        drug2ingredient_list.append(row)
# drug code, active_ing_code, ingredient, ingredient_supplied_ind, strength, .....
drug2ingredient_list = np.array(drug2ingredient_list[1:])
drug2ingredient = {}
for row in drug2ingredient_list:
    if row[0] != drug2ingredient:
        drug2ingredient[row[0]] = row[2]
    else:  # one drug may includes more than one ingredients, refering to multiple cols
        drug2ingredient[row[0]] += '/' + row[2]
def queryDIN(din):
    try:
        return drug2ingredient[din2drug[din]]
    except:
        return 'Not Found' 



def draw_heatmap(mode, beta, words, topic_code, topic_ref, topics_id, ax):
    if topics_id.any() != None:
        beta = beta[topics_id, :]
        draw_words = np.array(words, dtype=object)[topics_id]
    else:
        draw_words = words
    # draw_topic_code = np.array(topic_code)[topics_id, :]
    # draw_topic_ref = np.array(topic_ref)[topics_id, :]
    ws, ids = np.unique([w for wrow in draw_words for w in wrow ], return_index=True)
    unique_words = ws[np.argsort(ids)]
    code = [vocab[a] for a in unique_words]
    code_ref = [query_code(mode, i) for i in code]

    beta_data = pd.DataFrame(data=beta[:, unique_words].T, columns=topics_id, index=['-'.join(p) for p in zip(code, code_ref)])
    # beta_data = pd.DataFrame(data=beta[:,unique_words].T, columns=np.arange(len(beta))+1, index=[p for p in ICD])
    sns.heatmap(beta_data, xticklabels=0, vmax=0.2, center=0, cmap="RdBu_r", ax=ax)
    if (len(topics_id) > 10):
        ax.set_title(path+mode+' # (topic): ' + str(len(topics_id)))
    else:
        ax.set_title(path+mode+'topic: ' + ','.join([str(t) for t in list(topics_id)]))
    plt.show()


def query_code(mode, code):
    if mode == 'icd':
        return queryICD(code)
    elif mode == 'act_code':
        return queryACT(code)
    else:
        return queryDIN(code)
    
def load_beta(mode, path):
    files = os.listdir(path)
    for i in files: 
        if mode in i and 'beta' in i:
            print(i)
            beta = scipy.io.loadmat(path+i)['values']
    print("# (topics): " + str(len(beta)))
    return beta

def get_topic_words(mode, vocab, beta, k=5, return_weight=False):
    print('get top words...')
    words = []
    weights = []
    for i, b in enumerate(beta):
        idb = np.argsort(b)[-k:][::-1]
        words_ = idb
        words.append(words_)
        weights.append(b[idb])
    topic_code = [[vocab[w] for w in t_w] for t_w in words]
    # print('topic diseases: ')
    topic_code_ref = [[mode[0]+v + '-'+query_code(mode,v)[:30] for v in t_w] for t_w in topic_code]

    if return_weight:
        return (topic_code, topic_code_ref), weights
    else:
        return words, topic_code, topic_code_ref


def render(vocab, path, mode, coderef_writer, ax, save=False):
    beta = load_beta(mode, path)
    words, topic_code, topic_code_ref = get_topic_words(mode, vocab, beta)
    K = len(beta)

    if save:
        with open(path+'topic_'+mode+'_info.pkl', 'wb') as f:
            pickle.dump({mode: topic_code, 'disease': topic_code_ref}, f)
        if coderef_writer != None:
            pd.DataFrame(data=np.array(topic_code_ref), index=np.arange(K)).to_excel(coderef_writer, sheet_name=mode, engine='xlsxwriter')
        with open(path + 'topic_ref_'+mode+'.txt', 'w') as f:
            for i, c_r in enumerate(topic_code_ref):
                print((i, c_r), file=f)
        with open(path + 'topic_'+mode+'.txt', 'w') as f:
            for w in topic_code:
                print(w, file=f)

    ws, ids = np.unique([w for wrow in topic_code for w in wrow ], return_index=True)
    unique_codes = ws[np.argsort(ids)]
    code = unique_codes
    code_ref = [query_code(mode, i) for i in code]
    print("# (tokens): " + str(len(unique_words)))
    print(mode)
    print(code)


    # topics_id = np.sort(np.random.choice(topics_candidates, size=5, replace=False))
    # topics_id = [135, 156, 56, 37, 167, 166]   mortality
    # topics_id = [135, 83, 123, 167, 147, 166]   age (old)
    # print(topics_id)
    topics_id = np.arange(K)
    draw_heatmap(mode, beta, words, code, code_ref, topics_id, ax)

def type_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    if word[0] == 'i':
        return (255,0,0)
    elif word[0] == 'a':
        return (177, 173,41)
    else:
        return (0,0,255)
def draw_topic_word_cloud(vocab, path, mode, topics, axes):
    # topics = np.random.choice(np.arange(200), size=9, replace=False)
    beta = load_beta(mode, path)
    (_, topic_code_ref), w = get_topic_words(mode, vocab, beta[topics], k=5, return_weight=True)
    print(topic_code_ref[0])

    #topic_code_ref_trunc = [[ref[:30] for ref in topic_ref] for topic_ref in topic_code_ref]
    # topic_code_ref = topic_code_ref_trunc 
    
    # print(topic_code_ref)
    for t, t_ref, t_w, ax in zip(topics, topic_code_ref, w, axes):
        t_w /= sum(t_w)
        ax.imshow(WordCloud(contour_width=1, prefer_horizontal=1, background_color='white',color_func=type_color_func).fit_words(dict(zip(t_ref, t_w))))
        ax.title.set_text('topic ' + str(t)+' - '+mode)
        ax.axis('off')


    




# path = 'embed/out-patient/collect_ETM/';  K = 200
# path = 'embed/out-patient/DETM/COPD_K10_PPL1178/'
# path = 'embed/out-patient/COPD/COPD_etm_K20_PPL532/'; K = 20

# path = 'embed/out-patient/COPD/newCOPD_age20_K5_PPL380/'; K = 5
# path = 'embed/out-patient/COPD/newCOPD_age30_K5_PPL385/'; K = 5
# path = 'embed/out-patient/COPD/COPD_ETMrho_PPL407/'; K = 5
path = 'embed/out-patient/COPD/corCOPD_ETMrho_PPL364/'; K = 5



multi_modal = ['icd', 'act_code', 'din']
vocab_ = {}
for mode in multi_modal:
    vocab_[mode] = pickle.load(open('data/COPD_out-patient/'+mode+'_vocab.pkl', 'rb'))

# f, axes = plt.subplots(1, len(multi_modal))
# with ExcelWriter(path+'code_ref.xlsx') as coderef_writer:
#     for i, mode in enumerate(multi_modal):
#         vocab = pickle.load(open('data/out-patient/'+mode+'_vocab.pkl', 'rb'))
#         render(vocab, path, mode, coderef_writer, axes[i])


# f, axes = plt.subplots(1, 3)
# axes = axes.flatten()
# for i, mode in enumerate(multi_modal):
#     vocab = pickle.load(open('data/out-patient/'+mode+'_vocab.pkl', 'rb'))
#     render(vocab, path, mode, None, None, save=False)


def draw_topics_across_ages():
    # mode = multi_modal[0]
    f, axes_ = plt.subplots(8, 3)
    # draw_topic_word_cloud(vocab, path, mode, topics=[135, 156, 56, 37, 167, 166])  # mortality
    topics_split = np.split(np.arange(200), int(200/5))
    topics_id_mor = [[135, 156, 56], [37, 167, 166]]   #  mortality
    topics_id_age = [[135, 83, 123], [167, 147, 166]]   # age (old)
    topics_split = topics_id_mor +topics_id_age
    topics_split = [[135,  83, 156], [135, 156,  54], [ 56, 156,  54], [100, 194, 156], [150, 100, 147], [150, 147, 166], [166, 150,  10], [166,  10,  37]]   # age
    f, axes_ = plt.subplots(len(topics_split), 3)
    plt.gcf().set_size_inches(11,18)
    age_ranges = ['0~10', '10~20', '20~50', '50~70', '70~80', '80~90', '90~100', '>100']
    for topics, axes, age_range in zip(topics_split, axes_, age_ranges):
        topic_code_ref_s = []
        w_s = []
        for c, mode in enumerate(multi_modal):
            # draw_topic_word_cloud(vocab[mode], path, mode, topics=topics, axes=axes[:,c])  # age
            # embed()
                # topics = np.random.choice(np.arange(200), size=9, replace=False)
            vocab = vocab_[mode]
            
            beta = load_beta(mode, path, temporal=True)
            (_, topic_code_ref), w = get_topic_words(mode, vocab, beta[topics], k=5, return_weight=True)

            #topic_code_ref_trunc = [[ref[:30] for ref in topic_ref] for topic_ref in topic_code_ref]
            # topic_code_ref = topic_code_ref_trunc 
            topic_code_ref_s.append(topic_code_ref)
            w /= sum(w)
            w_s.append(w)
        topic_code_ref_s = np.hstack(topic_code_ref_s)
        w_s = np.hstack(w_s)
            
            # print(topic_code_ref)
        for t, t_ref, t_w, ax in zip(topics, topic_code_ref_s, w_s, axes):
            # t_w /= sum(t_w)
            ax.imshow(WordCloud(contour_width=1, prefer_horizontal=1, background_color='white',color_func=type_color_func, relative_scaling=1).fit_words(dict(zip(t_ref, t_w))))
            ax.title.set_text('topic ' + str(t))
            ax.axis('off')
        axes[0].set_ylabel(age_range)
        plt.suptitle('Topic: '+str(topics))
        # plt.show()
        # plt.savefig('WC_'+str(topics), format='jpg', dpi=600)
    plt.suptitle('Topic trend develop when at different age')
    plt.savefig('WC_age', format='jpg', dpi=1200)

# mode = multi_modal[0]
# draw_topic_word_cloud(vocab, path, mode, topics=[135, 156, 56, 37, 167, 166])  # mortality
# topics_split = [[135,  83, 156], [135, 156,  54], [ 56, 156,  54], [100, 194, 156], [150, 100, 147], [150, 147, 166], [166, 150,  10], [166,  10,  37]]   # age
# f, axes_ = plt.subplots(len(topics_split), 3)
# plt.gcf().set_size_inches(11,18)
for c, mode in enumerate(multi_modal):
    beta = load_beta(mode, path)
    f = open(os.path.join(path, 'topic_'+mode+'.txt'), 'w') 

    topics_code_ref_all = []
    w_all = []
    # For ETM
    # (_, topics_code_ref_t), w = get_topic_words(mode, vocab_[mode], beta, k=5, return_weight=True)
    # for i in range(K):
    #     print((i, topics_code_ref_t[i]), file=f)

    # For DETM
    T = 8
    for t in range(T):
        (_, topics_code_ref_t), w = get_topic_words(mode, vocab_[mode], beta[t], k=5, return_weight=True)
        topics_code_ref_all.append(topics_code_ref_t)
        w_all.append(w)
    w_all = np.array(w_all)
    for i in range(K):
        # FIXME: scatter plot
        # plt.clf()
        # y = w_all[:,i,:].flatten()
        # x = np.hstack([[t] * 5 for t in range(T)]) + 3
        # cc = np.hstack([topics_code_ref_all[t][i] for t in range(T)])
        # u, ind = np.unique(cc, return_index=True)
        # cc_unique = u[np.argsort(ind)]
        # color_map = dict(zip(cc_unique, cm.rainbow(np.linspace(0, 1, len(cc_unique)))[::-1]))
        # c = list(map(color_map.get, cc))
        # g = sns.stripplot(x=x,y=y, hue=cc)
        # g.legend_.remove()
        # sns.despine()
        # plt.show()
        # embed()
        
        for t in range(T):
            print((i, t*10+30, topics_code_ref_all[t][i]), file=f)
    f.close()


alpha = scipy.io.loadmat(path+'alpha/_alpha_epoch20.mat')['values']
alpha = alpha[:-1,:,:]

# Topic embeddings alpha
from matplotlib.pyplot import gcf
L = alpha.shape[-1]
data = np.vstack([alpha[:,i,:].T for i in range(K)])
# data = pd.DataFrame(data=data)

age_label = [str((i+3)*10)+'-'+str((i+4)*10) for i in range(T-1)] + ['100-']
# age_label = np.arange(T)
data = pd.DataFrame(data, columns=age_label)

topic_color = sns.cubehelix_palette(K)
age_color = cm.rainbow(np.linspace(0, 1, T)[::-1])
topic_cmap = dict(zip(np.arange(K), topic_color))
age_cmap = dict(zip(age_label, age_color))
cc1 = pd.Series([int(i/L) for i in range(K*L)], name='topic').map(topic_cmap)
cc2 = pd.Series(age_label, name='age', index=age_label).map(age_cmap)

g = sns.clustermap(data, cmap='RdBu_r', col_colors=cc2, row_colors=cc1, 
    col_cluster=False, row_cluster=False,
    xticklabels=True, yticklabels=False,
    dendrogram_ratio=0.12,
    cbar_pos=(0.02, 0.2, 0.05, 0.6))


for label, color in zip(age_label, age_color): 
    g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
# l1 = g.ax_col_dendrogram.legend(title='Age', loc="center", ncol=1, bbox_to_anchor=(0.1, 0.8), bbox_transform=gcf().transFigure)

topic_label = [str(i) for i in range(K)]
for label, color in zip(topic_label, topic_color): 
    g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
l = g.ax_col_dendrogram.legend( loc="center", ncol=7, bbox_to_anchor=(0.55, 0.93), bbox_transform=gcf().transFigure)

plt.show()

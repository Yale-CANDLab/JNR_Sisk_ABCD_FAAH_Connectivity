#!/usr/bin/env python
# coding: utf-8

# # FAAH Genotype Analysis

# In[1]:


# Set variables and paths

import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
# from statannot import add_stat_annotation
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.factorplots import interaction_plot
from scipy.stats import ttest_ind
from datetime import date
today = str(date.today())
import bct
import sys
from glob import glob

var1 = sys.argv[1]
print("iter number is {}".format(var1))

#Read in Rest data
home = '/home/lms233/ABCD_FAAH'
out_path = '/gpfs/milgram/scratch60/gee_dylan/lms233/ABCD_FAAH/fitted_models_5.2021'
baseline = pd.read_csv(home + '/Baseline_Masterfile_n=3266_2021-05-21.csv')
#Set model variables
yvar = 'cbcl_scr_dsm5_anxdisord_r'
#Set paths
all_mats_dir = '/home/lms233/ABCD_FAAH/rest_matrices_averaged'
all_behav_dir = all_mats_dir

var1 = sys.argv[1]
print("iter number is {}".format(var1))


# In[14]:


def prepare_data():
    #Glob list of all files in this dir that match string
    subfiles = glob(all_mats_dir + '/sub-NDAR*.csv')

    #Create list of subjectids
    subs = []
    for i in range(0, len(subfiles)):
        s = subfiles[i]
        x = s.replace(all_mats_dir + '/', '')
        sub = x.split('_')[0]
        subs.append(sub)

    #Rename subjectkey 
    subsers = pd.Series(subs, name = 'subjectkey').str.replace('sub-NDAR', 'NDAR_')

    #Merge with 'selected' dataframe (thus excluding subs that did not pass QC) 
    data1 = pd.merge(subsers, baseline, how = 'inner', on='subjectkey')
    pre_nbs_data = data1[data1.cbcl_eventname  == 'baseline_year_1_arm_1'] #'''follow_up_y_arm_1'] #

    assert len(pre_nbs_data) == len(baseline.dropna())

    type1 = pre_nbs_data[pre_nbs_data.genotype == 1].drop('level_0', axis=1).reset_index()
    type2 = pre_nbs_data[pre_nbs_data.genotype == 2].drop('level_0', axis=1).reset_index()
    
    return pre_nbs_data, type1, type2


# In[15]:


pre_nbs_data, type1, type2 = prepare_data()


# In[16]:


#Function to read in all data
def read_in_nbs_data(type1, type2):
    #Create empty matrices to append data into
    mat_1 = np.zeros(shape=(368, 368, len(type1)))
    mat_2 = np.zeros(shape=(368, 368, len(type2)))

#     assert mat_1.shape[2] + mat_2.shape[2] == len(baseline)
    # Read in data for type 1
    print('Reading in subject data ({} subjects)'.format(len(type1)))
    for x in range(0, len(type1)):
        #Set subject
        sub = type1['subjectkey'][x].replace('NDAR_',  'sub-NDAR')
        try:
            # Read in data and append in order (3rd dim)
            #print("Working on {}, {} out of {}".format(sub, x, len(type1)))
            rest_file = glob(all_mats_dir + '/{}*.csv'.format(sub))[0]
            mat = pd.read_csv(rest_file, header=0, index_col=0, engine='python').to_numpy()[:,0:368]
            mat_1[:,:,x] = mat
        except:
            print("Error on {}".format(sub))

    # Read in data for type 2
    print('Reading in subject data ({} subjects)'.format(len(type2)))
    for x in range(0, len(type2)):
        #Set subject
        sub = type2['subjectkey'][x].replace('NDAR_',  'sub-NDAR')
        try:
            # Read in data and append in order (3rd dim)
            #print("Working on {}, {} out of {}".format(sub, x, len(type2)))
            rest_file = glob(all_mats_dir + '/{}*.csv'.format(sub))[0]
            mat = pd.read_csv(rest_file, header=0, index_col=0, engine='python').to_numpy()[:,0:368]
            mat_2[:,:,x] = mat
        except:
            print("Error on {}".format(sub))
    return mat_1, mat_2


# In[17]:


# Filter mask to remove duplicate indices
def filter_mask(adj_mat, val):
    mask = np.argwhere(adj_mat == val)
    mask = mask.tolist()
    final_mask = []
    for i in range(0, len(mask)):
        pair = mask[i]
        v1 = pair[0]
        v2 = pair[1]
        if [v2, v1]  in final_mask:
            pass
        else:
            final_mask.append([v1, v2])
    print("Original mask length: {}".format(len(mask)))
    print("Length with duplicates removed: {}".format(len(final_mask)))
    return final_mask
    
#network_mask = filter_mask(nbs_adj, network_value)   


# In[18]:


#Function to extract values
def get_network_values(mask, mats, subs, valence):
    n_pairs = int(len(mask))
    mask_mat = np.ones((mats.shape[2], n_pairs))
    for i in range(0, n_pairs):
        inds = mask[i]
        x = inds[0]
        y = inds[1]
        mask_mat[:, i]= mats[x, y]
    net_df = pd.DataFrame(mask_mat)
    net_df_mean = net_df.sum(axis=1)
    net_s = pd.Series(net_df_mean, name = "net_{}_var".format(valence))
    return net_s, net_df

#Function to separate left and right tails
def get_tails(mask, ctl_mat, pt_mat):
    n_pairs = int(len(mask))
    positive_edges = []
    negative_edges = []
    for i in range(0, len(mask)):
        inds = mask[i]
        x = inds[0]
        y = inds[1]
        col1= ctl_mat[x, y]
        col2= pt_mat[x, y]
        t, p = ttest_ind(col1, col2)
        if t > 0:
            #print('{} > than {}'.format(col1.mean().round(3), col2.mean().round(3))) (Sanity check)
            negative_edges.append(inds)
        if t < 0:
            #print('{} < than {}'.format(col1.mean().round(3), col2.mean().round(3))) (Sanity check)
            positive_edges.append(inds)
    return positive_edges, negative_edges

def get_amyg_coords(tail):
    coord_df = pd.DataFrame(tail)  
    coords = []
    for i in range(0, len(coord_df)):
        row = coord_df.iloc[i, :]
        if row[0] == 332:
            coords.append(row)
        elif row[1] == 332:
            coords.append(row)
        elif row[0] == 339:
            coords.append(row)
        elif row[1] == 339:
            coords.append(row)
        else:
            pass
    amyg_mask = pd.DataFrame(coords).to_numpy()
    return amyg_mask

def get_amyg_mat(amyg_mask):
    #Create amygdala adjacency matrix
    amyg_adj_mat = np.zeros([368, 368])
    for i in range(0, len(amyg_mask)):
        val1 = amyg_mask[i, 0]
        val2 = amyg_mask[i, 1]
        amyg_adj_mat[val1, val2] = 1.0
        amyg_adj_mat[val2, val1] = 1.0
    return amyg_adj_mat

#Function to extract values -- double checked this and it is correct
def get_amyg_values(mask, mats, subs, name):
    n_pairs = int(len(mask))
    mask_mat = np.ones((mats.shape[2], n_pairs))
    for i in range(0, n_pairs):
        inds = mask[i]
        x = inds[0]
        y = inds[1]
        mask_mat[:, i]= mats[x, y]
    net_df = pd.DataFrame(mask_mat)
    net_df_mean = net_df.sum(axis=1)
    net_s = pd.Series(net_df_mean, name = "net_{}_var".format(name))
    return net_s, net_df


# In[25]:


#Run Network-Based Statistic, use 10-fold cross validation

def run_analysis(n_CVsplits, data, nbs_kval, nbs_threshold, iter_number):
    from sklearn.model_selection import StratifiedKFold
    from datetime import date
    today = str(date.today())
    
    #$et variables
    n_splits = n_CVsplits
    X = data[['subjectkey', 'genotype']]
    y = data['genotype']
    nbs_k = nbs_kval #number of NBS permutations per fold
    nbs_thresh = nbs_threshold #Threshold NBS at p=0.05 (1.96); p=0.01 (2.58)
    nbs_mats = np.ones((368, 368, n_splits+1))
    nbs_pvals = []
    j = iter_number
    
    #Set kfold variables
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=j)
    skf.get_n_splits(X, y)
    count = []
    for train_index, test_index in skf.split(X, y):
        count.append(1)
        fold = len(count)
        train_subs = X.loc[train_index]
        assert len(train_subs) == len(train_index)
        train_cc = train_subs[train_subs['genotype'] == 1].reset_index()
        train_a = train_subs[train_subs['genotype'] == 2].reset_index()

        test_subs = X.loc[test_index]
        test_cc = test_subs[test_subs['genotype'] == 1].reset_index()
        test_a = test_subs[test_subs['genotype'] == 2].reset_index()
        assert len(test_subs) == len(test_index)

        #Read in data
        train_mat_c, train_mat_a = read_in_nbs_data(train_cc, train_a)
        test_mat_c, test_mat_a = read_in_nbs_data(test_cc, test_a)

        #Run NBS on training data
        nbs_pval, nbs_adj, nbs_null = bct.nbs_bct(x = train_mat_c, y = train_mat_a, 
                                              thresh = nbs_thresh, k=nbs_k, #Thresholded at t= 1.96, p=0.05 (t= 2.58; p=0.01) according to calculation here (df 2128; for two-tailed) https://statscalculator.com/tcriticalvaluecalculator?x1=0.01&x2=2128
                                              tail='both', paired=False, verbose=False)

        nbs_mats[:,:, len(count)] = nbs_adj
        nbs_pvals.append(nbs_pval)
        today = str(date.today())
        pd.DataFrame(nbs_adj).to_csv(home + '/NBS_AdjMat_{}_{}_{}.pkl'.format(fold, iter_number, today))
        #Clean adjacency matrix
        if 5 in nbs_adj:
            network_value = 5
        elif 4 in nbs_adj:
            network_value = 4
        elif 3 in nbs_adj:
            network_value = 3
        elif 2 in nbs_adj:
            network_value = 2
        else:
            network_value = 1
        print("NBS network value: {}".format(network_value))   
        network_mask = filter_mask(nbs_adj, network_value) 

        #### SET VARIABLES FOR MIXED MODELS
        mat_1 = test_mat_c
        mat_2 = test_mat_a

        t1subs = test_cc['subjectkey']
        t2subs = test_a['subjectkey']

        #Filter into positive and negative tails for TESTING dataset
        pos_tail, neg_tail = get_tails(network_mask, mat_1, mat_2)
        print(len(pos_tail), len(neg_tail))

        #Get positive and negative tails as dataframes for subs
        pos_network_t1, pos_network_t1_df = get_network_values(pos_tail, mat_1, t1subs, 'pos')
        pos_network_t2, pos_network_t2_df = get_network_values(pos_tail, mat_2, t2subs, 'pos')
        print(pos_network_t1.shape, print(pos_network_t2.shape))

        neg_network_t1, neg_network_t1_df = get_network_values(neg_tail, mat_1, t1subs, 'neg')
        neg_network_t2, neg_network_t2_df = get_network_values(neg_tail, mat_2, t2subs, 'neg')
        print(neg_network_t1.shape, print(neg_network_t2.shape))

        # Concatenate networks into one dataframe
        t1_nets = pd.concat([t1subs, pos_network_t1, neg_network_t1], axis=1)
        t2_nets = pd.concat([t2subs, pos_network_t2, neg_network_t2], axis=1)
        all_nets = pd.concat([t1_nets, t2_nets], axis=0)
        all_nets['subjectkey'] = all_nets['subjectkey'].str.replace('sub-NDAR', 'NDAR_')

        #Merge with Selected DF 
        all_df = pd.merge(baseline, all_nets, on = 'subjectkey', how = 'inner')
        all_df['genotype'] = all_df['genotype'].astype('category')
        assert len(all_df) == len(t1subs) + len(t2subs)

        all_df['ethnicity'] =all_df['ethnicity'].astype('category')
        all_df['fam_income'] = all_df['fam_income'].astype('category')
        all_df['mat_ed'] = all_df['mat_ed'].astype('category')
        all_df['sex'] = all_df['sex'].astype('category')
        data = all_df.drop(['level_0', 'index'], axis=1).reset_index()

        #Confirm genotype predicts network conn 
        today = str(date.today())

        data['sex'].cat.remove_unused_categories(inplace = True)
        data['fam_income'].cat.remove_unused_categories(inplace = True)
        data['mat_ed'].cat.remove_unused_categories(inplace = True)

        try:
            mod1 = sm.MixedLM.from_formula("net_pos_var ~ genotype + cbcl_age + sex +pds + fam_income + mat_ed + ethnicity + genetic_af_african + genetic_af_european + genetic_af_east_asian + genetic_af_american + mean_ffd", 
                                             re_formula="1", vc_formula={"fam_id": "0 + C(rel_family_id)"},
                            groups="mri_info_deviceserialnumber", data=data)
            results1 = mod1.fit()
            results1.save(outpath + '/Folds/Fold{}_LMEOutput_GenotypeDiffs_PosTail_{}_{}.pkl'.format(fold, j, today))
            summary1 = results1.summary()
            print(summary1)
        except Exception as inst:
            print('Main model 1 failed')
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .arg
            print(inst)    
                
        try:
            mod2 = sm.MixedLM.from_formula("net_neg_var ~ genotype + cbcl_age + sex +pds + fam_income + mat_ed + ethnicity + genetic_af_african + genetic_af_european + genetic_af_east_asian + genetic_af_american+ mean_ffd", 
                                             re_formula="1", vc_formula={"fam_id": "0 + C(rel_family_id)"},
                            groups="mri_info_deviceserialnumber", data=data)
            results2 = mod2.fit()
            results2.save(outpath + '/Folds/Fold{}_LMEOutput_GenotypeDiffs_NegTail_{}_{}.pkl'.format(fold, j, today))
            summary2 = results2.summary()
            print(summary2)
        except Exception as inst:
            print('Main model 2 failed')
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .arg
            print(inst)  
            
        # Model interaction effects between network connectivity and genotype
        try:
            intmod1 = sm.MixedLM.from_formula("cbcl_anxdisord_r ~ net_pos_var * genotype + cbcl_age + sex +pds + fam_income + mat_ed + ethnicity + genetic_af_african + genetic_af_european + genetic_af_east_asian + genetic_af_american+ mean_ffd", 
                                             re_formula="1", vc_formula={"fam_id": "0 + C(rel_family_id)"},
                            groups="mri_info_deviceserialnumber", data=data)
            intresults1 = intmod1.fit()
            intresults1.save(outpath + '/Folds/Fold{}_LMEOutput_NetworkGenotypeInteraction_PosTail_{}_{}.pkl'.format(fold, j, today))
            intsummary1 = intresults1.summary()
            print(intsummary1)
        except Exception as inst:
                print('Interaction model 1 failed')
                print(type(inst))    # the exception instance
                print(inst.args)     # arguments stored in .arg
                print(inst)    
        
        try:
            intmod2 = sm.MixedLM.from_formula("cbcl_anxdisord_r ~ net_neg_var * genotype + cbcl_age + sex +pds + fam_income + mat_ed + ethnicity + genetic_af_african + genetic_af_european + genetic_af_east_asian + genetic_af_american+ mean_ffd", 
                                             re_formula="1", vc_formula={"fam_id": "0 + C(rel_family_id)"},
                            groups="mri_info_deviceserialnumber", data=data)
            intresults2 = intmod2.fit()
            intresults2.save(outpath + '/Fold{}_LMEOutput_NetworkGenotypeInteraction_NegTail_{}_{}.pkl'.format(fold, j, today))
            intsummary2 = intresults2.summary()
            print(intsummary2)
        except Exception as inst:
                print('Interaction model 1 failed')
                print(type(inst))    # the exception instance
                print(inst.args)     # arguments stored in .arg
                print(inst)    
            

run_analysis(n_CVsplits=10, data=pre_nbs_data, nbs_kval=100, nbs_threshold=2.58, iter_number=1)


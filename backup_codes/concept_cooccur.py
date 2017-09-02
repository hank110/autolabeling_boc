from collections import defaultdict, namedtuple
from operator import attrgetter
import math

import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize

def initialize_coocc_matrix(w2c_mapping_file):
    column_names=['word','concept']
    df=pd.read_csv(w2c_mapping_file,header=None,names=column_names)
    concept_groupby=df.groupby('concept')
    concept_counter=concept_groupby.count()
    concept_num=concept_counter.size
    w2c_hash, w2i_hash=create_index(concept_groupby,concept_num) 
    coocc_container=[]
    for i in range(concept_num):
        single_concept=concept_groupby.get_group(i)
        word_num=len(single_concept.index)
        coocc_container.append(np.zeros((word_num,word_num)))
    return coocc_container, w2c_hash, w2i_hash


def create_index(concept_groupby,concept_num):
    w2c_hash={}
    w2i_hash={}
    for i in range(concept_num):
        single_concept=concept_groupby.get_group(i)
        word_num_groupby=len(single_concept.index)
        for j in range(word_num_groupby):
            word=single_concept.iloc[j]['word']
            concept=single_concept.iloc[j]['concept']
            w2c_hash[word]=int(concept)
            w2i_hash[word]=j
    return w2c_hash, w2i_hash


def fill_coocc_matrix(document_path,coocc_container,w2c_hash,w2i_hash):
    with open(document_path, 'r') as f:
        for line in f:
            tokens=line.split()
            for i, w in enumerate(tokens):
                try:
                    concept_num=w2c_hash[w]
                    target_m=coocc_container[concept_num]
                    word1_index=w2i_hash[w]
                except KeyError:
                    continue
                for j, ct in enumerate(tokens): 
                    try:
                        context_concept_num=w2c_hash[ct]
                        word2_index=w2i_hash[ct]
                    except KeyError:
                        continue
                    if (i>=j or concept_num != context_concept_num) : continue
                    elif (word1_index==word2_index): continue
                    else:
                        target_m[word1_index][word2_index]+=1
                        target_m[word2_index][word1_index]+=1
    return coocc_container  


def cal_sparsity(row_vector, t):
    '''
    Greater the calculated value, more sparse the vector
    '''
    k=len(row_vector)
    bottom=(math.pow(k,1)/math.pow(k,1/2))-1
    top=(math.pow(k,1)/(math.pow(k,1/2)+0.000000000000000000001))-(LA.norm(row_vector,1)/(LA.norm(row_vector,2)+0.00000000000000001))
    if t=='yes':
        print(bottom)
        print(top)
    return top/(bottom+0.000000000000000000000000000001)


def get_wordsinconcept(w2c_hash, concept_num):
    word_container=[]
    for word, concept in w2c_hash.items():
        if concept==concept_num:
            word_container.append(word)
    return word_container


def reconstruct_i2whash(word_list, w2i_hash):
    i2w_hash={}
    for word in word_list:
        i2w_hash[w2i_hash[word]]=word
    return i2w_hash


def aggr_sparsity(coocc_container, w2c_hash, w2i_hash):
    final_result=[]
    concept_num=0
    for ematrix in coocc_container:
        print(concept_num)
        word_container=get_wordsinconcept(w2c_hash, concept_num)
        i2w=reconstruct_i2whash(word_container,w2i_hash)
        each_result=[]
        word_index=0
        for erow in ematrix:
            word_sparsity=namedtuple('word_sparsity','word sparsity')
            each_result.append(word_sparsity(word=i2w[word_index], sparsity=cal_sparsity(erow, t='no')))
            if(i2w[word_index]=='output'): print(cal_sparsity(erow)*100, t='yes')
            if(i2w[word_index]=='protest'): 
                print(i2w[word_index])
                print(cal_sparsity(erow)*100, t='yes')
            word_index+=1
        final_result.append(sorted(each_result, key=attrgetter('sparsity')))
        concept_num+=1
    return final_result


def save_result(output_path, w2c_hash, sparsity_list):
    with open(output_path, 'w') as f:
        for econcept in sparsity_list:
            for escore in econcept:
                f.write("%s,%s,%s \n" %(escore.word, str(w2c_hash[escore.word]), str(escore.sparsity)))


def cal_wordsparsity_per_concept(training_doc, w2c_mapper, output_path):
    container, w2c, w2i = initialize_coocc_matrix(w2c_mapper)
    print("Initialization completed")
    a=fill_coocc_matrix(training_doc, container, w2c, w2i)
    print("Matrix creation completed")
    save_result(output_path,w2c,aggr_sparsity(a,w2c,w2i))


def main():
    w2c_mapper='../trained_results/w2c_d100_w8_mf50_c100.csv'
    document='/home/hank/Backup_data/data/finalex_reuters-cleaned-document_without_zeros.txt'
    output='d100_c100_results.csv'
    cal_wordsparsity_per_concept(document, w2c_mapper, output)

if __name__=="__main__":
    main()

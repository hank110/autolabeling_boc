# Word2Vec Training Setup
## input: training document (Need to arrange the doc such that each line = one sentence in txt)
## parameters: embedding dimension, window size, minimum word frequency threshold

document='/data/finalex_reuters-cleaned-document_without_zeros.txt'
dimensions=[100,200,300]
context=8				
min_freq=50
num_concepts=[100,200,300]						

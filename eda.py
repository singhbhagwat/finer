import datasets
from datasets import Dataset
import pandas as pd
import numpy as np

def EDA():
          
    train_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='train')

    dataset_tags = train_dataset.features['ner_tags'].feature.names

    tag2idx = {tag: int(i) for i, tag in enumerate(dataset_tags)}
    print('tag2idx:\n', tag2idx)

    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    print('idx2tag:\n', idx2tag)

    trainDF = train_dataset.to_pandas()
    print(trainDF.columns)

    print('************ XBRL Tag Frequency *********\n')
    df = trainDF.ner_tags.explode().value_counts().reset_index()
    df.columns = ['id', 'count']
    df.loc[:, 'tag'] = df.loc[:, 'id'].apply(lambda x: idx2tag[x])
    df.to_csv('overall_tag_freq.csv', index=False, header=True)


def overlay_hierarchy():
  
    train_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='train')    
    
    dataset_tags = train_dataset.features['ner_tags'].feature.names
    print("dataset_tags:\n", dataset_tags, '\n')

    tag2idx = {tag: int(i) for i, tag in enumerate(dataset_tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    rq1_preprocess = pd.read_excel("RQ1_Preprocess.xlsx")
    rq1_tag_changes = rq1_preprocess.loc[rq1_preprocess.ParentLabelSame == 'No', ['Entity Label', 'Parent Label', 'B-ParentTagId', 'I-ParentTagId']]
    rq1_tag_changes.loc[:, 'B-Entity-Label'] = rq1_tag_changes.loc[:, 'Entity Label'].apply(lambda l: 'B-'+l)
    rq1_tag_changes.loc[:, 'I-Entity-Label'] = rq1_tag_changes.loc[:, 'Entity Label'].apply(lambda l: 'I-'+l)
    rq1_tag_changes.loc[:, 'B-ParentTagId'] = rq1_tag_changes.loc[:, 'B-ParentTagId'].astype(int)
    rq1_tag_changes.loc[:, 'I-ParentTagId'] = rq1_tag_changes.loc[:, 'I-ParentTagId'].astype(int)
    rq1_B_tag2idx = dict(zip(list(rq1_tag_changes.loc[:, 'B-Entity-Label']), list(rq1_tag_changes.loc[:, 'B-ParentTagId'])))
    rq1_I_tag2idx = dict(zip(list(rq1_tag_changes.loc[:, 'I-Entity-Label']), list(rq1_tag_changes.loc[:, 'I-ParentTagId'])))
    rq1_tag2idx = rq1_B_tag2idx | rq1_I_tag2idx

    rq1_tag_changes.loc[:, 'B-Parent-Label'] = rq1_tag_changes.loc[:, 'Parent Label'].apply(lambda l: 'B-'+l)
    rq1_tag_changes.loc[:, 'I-Parent-Label'] = rq1_tag_changes.loc[:, 'Parent Label'].apply(lambda l: 'I-'+l)
    B_tag2idx_rq1 = dict(zip(list(rq1_tag_changes.loc[:, 'B-Parent-Label']), list(rq1_tag_changes.loc[:, 'B-ParentTagId'])))
    I_tag2idx_rq1 = dict(zip(list(rq1_tag_changes.loc[:, 'I-Parent-Label']), list(rq1_tag_changes.loc[:, 'I-ParentTagId'])))
    tag2idx_rq1 = B_tag2idx_rq1 | I_tag2idx_rq1
    tag2idx_rq1.update({k: tag2idx[k] for k in tag2idx if k not in rq1_tag2idx})
    idx2tag_rq1 = {tag2idx_rq1[k]:k for k in tag2idx_rq1}
    (pd.DataFrame(tag2idx_rq1, index=['id']).T).to_csv("tag2idx_rq1.csv")
    (pd.DataFrame(idx2tag_rq1, index=['tag']).T).to_csv("idx2tag_rq1.csv")

    additional_tags_added = list(rq1_tag_changes.loc[:, 'B-Entity-Label']) + list(rq1_tag_changes.loc[:, 'I-Entity-Label'])
       
    #print('rq1_B_tag2idx:\n', rq1_B_tag2idx)
    #print('rq1_I_tag2idx:\n', rq1_I_tag2idx)
    #print('rq1_tag2idx:\n', rq1_tag2idx)

    trainDF = train_dataset.to_pandas()
    trainDF.loc[:, 'ner_tags'] = trainDF.ner_tags.apply(lambda l: np.array([ rq1_tag2idx[idx2tag[i]] if (idx2tag[i] in rq1_tag2idx) else i for i in l ]))

    (trainDF.loc[:, ['ner_tags']].tail(10000)).to_excel("test_rq1_id_changes.xlsx")

    train_dataset_rq1 = Dataset.from_pandas(trainDF)

    return (train_dataset, train_dataset_rq1)

overlay_hierarchy()

    

    

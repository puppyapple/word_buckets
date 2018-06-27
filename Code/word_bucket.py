# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import jieba
import jieba.analyse
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler

def simple_minmax(df, col, min_v=0, max_v=1):
    target = df[col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(min_v, max_v))
    scaler.fit(target)
    df[col] = scaler.transform(target)
    return df.sort_values(by=col, ascending=False)

# 根据已有标签的词频统计
def bucket_by_tag(bucket_level, comp_ctaglinks, comp_tag_merged):
    level_groups = comp_ctaglinks.groupby("level")
    comp_link = level_groups.get_group(bucket_level)
    comp_link_with_all_tags = comp_link.merge(comp_tag_merged, how="left", left_on="comp_id", right_on="comp_id")
    
    comps_total_num = len(set(comp_tag_merged.comp_id))
    idf_each_tag = comp_tag_merged.groupby("label_name").agg({"comp_id": "count"}).reset_index().apply({"label_name": lambda x: x, "comp_id": lambda x: np.log2(comps_total_num/(x + 1))}).rename(index=str, columns={"comp_id": "idf"})
    idf_dict = dict(zip(idf_each_tag.label_name, idf_each_tag.idf))
    
    tf_each_link = comp_link_with_all_tags.groupby(["link", "label_name"]).agg({"type":"count"}).groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index(level=1).rename(index=str, columns={"type": "tf"})
    tf_each_link["idf"] = tf_each_link.label_name.apply(lambda x: idf_dict.get(x, 0))
    tf_each_link["link_value"] = (tf_each_link.tf * tf_each_link.idf) #.apply(lambda x: np.log2(x))
    
    result_by_link = tf_each_link.groupby("link")
    link_list = list(result_by_link.groups.keys())
    result_df_dict = {link: simple_minmax(result_by_link.get_group(link)[["label_name", "link_value"]].copy(), "link_value") for link in link_list}
    link_result_dict = {k: dict(zip(v.label_name, v.link_value)) for k, v in result_df_dict.items()}
    tag_bucket_file_name = "../Data/Output/word_buckets/tag_bucket_lv%d.pkl" % bucket_level
    tag_bucket_file = open(tag_bucket_file_name, "wb")
    pickle.dump(link_result_dict, tag_bucket_file)
    tag_bucket_file.close()
    print("###tag_bucket_lv%d.pkl generated!###" % bucket_level)
    return link_result_dict



# 基于冠文本的统计，可以同时使用TF-IDF和Text_rank（jieba接口）
def bucket_by_text(bucket_level, comp_ctaglinks, comp_intro_merged):
    level_groups = comp_ctaglinks.groupby("level")
    comp_link = level_groups.get_group(bucket_level)
    comp_link_with_all_intros = comp_link.merge(comp_intro_merged, how="left", left_on="comp_id", right_on="comp_id")
    intro_merged_each_link = comp_link_with_all_intros.groupby("link").agg({"intro": lambda x: "。".join(set(x))})
    intro_merged_each_link["tf_idf"] = intro_merged_each_link.intro.apply(lambda x: jieba.analyse.tfidf(x, topK=None, withWeight=True))
    intro_merged_each_link["textrank"] = intro_merged_each_link.intro.apply(lambda x: jieba.analyse.textrank(x, topK=None, withWeight=True))
    intro_merged_each_link.tf_idf = intro_merged_each_link.tf_idf.apply(lambda x: dict(x))
    intro_merged_each_link.textrank = intro_merged_each_link.textrank.apply(lambda x: dict(x))
    
    tfidf_result_dict = dict(zip(intro_merged_each_link.index, intro_merged_each_link.tf_idf))
    textrank_result_dict = dict(zip(intro_merged_each_link.index, intro_merged_each_link.textrank))
    
    tfidf_bucket_file_name = "../Data/Output/word_buckets/tfidf_bucket_lv%d.pkl" % bucket_level
    textrank_bucket_file_name = "../Data/Output/word_buckets/textrank_bucket_lv%d.pkl" % bucket_level
    
    tfidf_bucket_file = open(tfidf_bucket_file_name, "wb")
    pickle.dump(tfidf_result_dict, tfidf_bucket_file)
    tfidf_bucket_file.close()
    
    textrank_bucket_file = open(textrank_bucket_file_name, "wb")
    pickle.dump(textrank_result_dict, textrank_bucket_file)
    textrank_bucket_file.close()
    
    return (tfidf_result_dict, textrank_result_dict)


def bucket_by_text_textrank(bucket_level, comp_ctaglinks, comp_intro_merged):
    level_groups = comp_ctaglinks.groupby("level")
    comp_link = level_groups.get_group(bucket_level)
    comp_link_with_all_intros = comp_link.merge(comp_intro_merged, how="left", left_on="comp_id", right_on="comp_id")
    intro_merged_each_link = comp_link_with_all_intros.groupby("link").agg({"intro": lambda x: "。".join(set(x))})
    intro_merged_each_link["textrank"] = intro_merged_each_link.intro.apply(lambda x: jieba.analyse.tfidf(x, topK=None, withWeight=True))
    return

def load_bucket(path):
    return pickle.load(open(path, "rb"))

def tagger(comp_word_list, bucket, top=5):
    link_list = list(bucket.keys())
    scores = [sum([bucket.get(link).get(word, 0) for word in comp_word_list]) for link in link_list]
    result = pd.DataFrame({"link_name": link_list, "score": scores})
    return result.sort_values(by="score", ascending=False)[:top]

def tagger_for_df(intro_df, bucket_list, col_name="words", top=5):
    if col_name not in intro_df:
        print("'words' column not found in dataframe!")
        return -1
    for name, bucket in bucket_list.items():
        intro_df[name] = intro_df.words.apply(lambda x: tagger(x, bucket, top=top).values)
    return intro_df
    
def multi_process_tagger(intro_df, bucket_list, col_name="words", top=5, process_num=8):
    if col_name not in intro_df:
        print("'words' column not found in dataframe!")
        return -1
    result_list = []
    split_intro_infos = np.array_split(intro_df, process_num)
    pool = mp.Pool()
    for i in range(0, process_num):
        result_list.append(pool.apply_async(tagger_for_df, (split_intro_infos[i], bucket_list,)))
    pool.close()
    pool.join()
    result_merged = pd.concat([r.get() for r in result_list])
    return result_merged

def bucket_generator(bucket):
    bucket_by_tag()
    bucket_by_text(0)
    return 0
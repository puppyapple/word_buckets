# -*- coding: utf-8 -*-
import pandas as pd
import configparser
import pickle

def comp_tag(data_raw_new, data_raw_old_full, nctag_filter_num=15):
    # 根据输入的公司概念和非概念标记源数据，分别得到完整的公司-概念标签、公司-非概念标签
    data_raw_new = data_raw_new[data_raw_new.remarks != "1"].copy()
    data_raw_new.dropna(subset=["comp_id", "label_name"], inplace=True)
    cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
    data_raw_new = data_raw_new[data_raw_new.label_name != ''][cols].copy()

    # comp_ctag_table_all_infos 是读取文件后带概念标签的全部信息
    comp_ctag_table_all_infos = data_raw_new[data_raw_new.classify_id != 4].reset_index(drop=True)
    comp_ctag_table = comp_ctag_table_all_infos[["comp_id", "label_name"]].reset_index(drop=True)
    
    # 新系统下结果中的公司-非概念标签
    data_raw_nctag_p1 = data_raw_new[data_raw_new.classify_id == 4][["comp_id", "label_name"]].copy()

    # 读取旧版数据，只取其中的非概念标签（概念标签无法确定其层级和产业链（复用））
    data_raw_old = data_raw_old_full[data_raw_old_full.key_word != ""][["comp_id", "key_word"]].copy()
    data_raw_old.dropna(subset=["comp_id", "key_word"], inplace=True)
    data_raw_old.columns = ["comp_id", "label_name"]

    # 命中了概念标签的公司全集列表，过滤掉旧版中未命中概念标签的公司，减小flatten计算量
    comps_with_ctag = comp_ctag_table_all_infos.comp_id.drop_duplicates()
    data_raw_old = data_raw_old[data_raw_old.comp_id.isin(comp_ctag_table_all_infos.comp_id.drop_duplicates())]
    
    # 新版的非概念标签和旧版整体数据拼接后进行split和flatten
    data_to_flatten = pd.concat([data_raw_old, data_raw_nctag_p1])
    tuples = data_to_flatten.apply(lambda x: [(x[0], t) for t in x[1].split(",") if t != ""], axis=1)
    flatted = [y for x in tuples for y in x]
    data_raw_nctag_flatted = pd.DataFrame(flatted, columns=["comp_id", "label_name"]).drop_duplicates()
    
    # 去掉已经是概念标签的记录作为公司-非概念标签的全集
    comp_nctag_table = data_raw_nctag_flatted[~data_raw_nctag_flatted.label_name.isin(comp_ctag_table_all_infos.label_name.drop_duplicates())].copy()
    
    # 合并成公司-标签的一一对应集合
    comp_ctag_table["type"] = "ctag"
    comp_nctag_table["type"] = "nctag"    
    
     # 过滤掉命中公司数太低的非概念标签
    nctag_aggregated = comp_nctag_table.groupby("label_name").agg({"type": "count"}).reset_index()
    comp_nctag_table = comp_nctag_table[comp_nctag_table.label_name.isin(nctag_aggregated[nctag_aggregated.type >= nctag_filter_num].label_name)]
    
    comp_tag_merged = pd.concat([comp_ctag_table, comp_nctag_table])
    
    # 生成公司id-name字典保存
    comp_id_name = pd.concat([data_raw_new[["comp_id", "comp_full_name"]], data_raw_old_full[["comp_id", "comp_full_name"]]]).drop_duplicates()
    comp_id_name_dict = dict(zip(comp_id_name.comp_id, comp_id_name.comp_full_name))
    comp_id_name_dict_file_name = "../Data/Output/word_buckets/comp_id_name_dict.pkl"
    comp_id_name_dict_file = open(comp_id_name_dict_file_name, "wb")
    pickle.dump(comp_id_name_dict, comp_id_name_dict_file)
    comp_id_name_dict_file.close()
    
    return (comp_tag_merged, comp_ctag_table_all_infos)


def link_cutter(link_str, sep):
    pieces = link_str.split(sep)
    return [(sep.join(pieces[:i]), i) for i in range(1, len(pieces) + 1)]
    
def comp_ctaglinks(comp_ctag_table_all_infos):
    # 公司和其命中的概念标签链
    to_filter = ["企业云服务", "企业大数据解决方案", "机器人本体", "行业大数据解决方案"]
    comp_ctag_table_all_infos.src_tags = comp_ctag_table_all_infos[["label_type_num", "src_tags"]].apply(lambda x: x[1].split("#")[x[0] - 1], axis=1)
    comp_ctag_table_all_infos_ = comp_ctag_table_all_infos[~comp_ctag_table_all_infos.src_tags.apply(lambda x: x.split("-")[0]).isin(to_filter)]
    comp_ctag_fulllink = comp_ctag_table_all_infos[["comp_id", "src_tags"]].drop_duplicates()
    comp_ctag_fulllink["link_pieces"] = comp_ctag_fulllink.src_tags.apply(lambda x: link_cutter(x, "-"))
    to_flatten = comp_ctag_fulllink.apply(lambda x: [(x[0], t[0], t[1]) for t in x[2]], axis=1)
    comp_ctaglinks = pd.DataFrame([y for x in to_flatten for y in x], columns=["comp_id", "link", "level"]).drop_duplicates()
    return comp_ctaglinks


#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-06-29 10:06
"""
# https://www.elastic.co/cn/
# Download and unzip Elastic search
# ./bin/elasticsearch  -d -p pid

import jieba, re, time, json
import jieba.analyse
import jieba.posseg as jbpseg
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch()

import_pos = ["a", "ad", "an", "b", "d", "h", "i", "j", "k", "l", "m", "n", "r", "nt", "nz", "vd", "vn", "v", "z", "un",
              "vg", "ng", "ag", "dg", ]


def create_index(index_name, mapping):
    """
    创建 索引，指定 索引名、索引结构
    """
    es.indices.create(index=index_name, body=mapping)
    print("create_es_index end...")


def add_docs(index_name):
    """
    Add documents to elastic search index.
    """
    t1 = time.time()
    did = 0
    for line in open("./baike_qa_valid.json"):
        if str(line.strip()) == "":
            continue
        did += 1
        qa_example = json.loads(line.strip())
        qid = qa_example["qid"]
        category = qa_example["category"].split("-")
        title = qa_example["title"]
        desc = qa_example["desc"]
        answer = qa_example["answer"]
        # jieba 词性过滤
        title_seg = [k for k, v in jbpseg.cut(title) if v in import_pos]
        desc_seg = [k for k, v in jbpseg.cut(desc) if v in import_pos]

        doc = {
            'qid': qid,
            'category': category,
            'title': title,
            'title_seg': " ".join(title_seg),
            'desc': desc,
            'desc_seg': " ".join(desc_seg),
            'answer': answer,
        }
        if did % 5000 == 0:
            t2 = time.time()
            print("cost time:", t2 - t1)
            print("did:", did)
            t1 = t2

        yield {
            '_index': index_name,
            # '_type': '_doc',
            '_source': doc,
        }


def es_query(query, index_name):
    """
    es 查询语句
    """
    s = es.search(index=index_name, body=query)
    return json.dumps(s, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    flag = False  # 没有初始化时，构建好 es ，并导入数据
    if flag:
        #  创建索引结构
        print("create index...")
        index_name_1 = "baike_qa"
        mapping_1 = {
            "mappings": {
                "properties": {
                    "qid": {
                        "type": "keyword"
                    },
                    "category": {
                        "type": "keyword"
                    },
                    "title": {
                        "type": "text"
                    },
                    "title_seg": {
                        "type": "text",
                        "analyzer": "whitespace"
                    },
                    "desc": {
                        "type": "text",
                    },
                    "desc_seg": {
                        "type": "text",
                        "analyzer": "whitespace"
                    },
                    "answer": {
                        "type": "text",
                    }
                }
            }

        }
        create_index(index_name_1, mapping_1)
        # 插入数据到es
        t1 = time.time()
        bulk(es, add_docs(index_name_1))
        t2 = time.time()
        print("cost time:", t2 - t1)

    # 1
    # query = {"query": {
    #     "match": {"category": "烦恼"}
    # }
    # }
    # text = " ".join(jieba.cut("请问深入骨髓地喜欢一个人怎么办我不能确定对方是不是喜欢我"))
    text = "请问深入骨髓地喜欢一个人怎么办我不能确定对方是不是喜欢我"
    query = {
        "query": {
            "bool": {
                "should": [
                    {"terms": {
                        "category": ["烦恼"]}},
                    {"match": {"title": {"query": text, "boost": 3}}},
                    # "boost": 8，#
                ]
            }
        }
    }
    a = es_query(query, "baike_qa")
    print(a)

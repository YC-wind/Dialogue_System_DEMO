#!/usr/bin/python
# coding:utf8
""" 
@version: v1.0 
@author: YC
@license: Apache Licence  
@contact: 990800269@qq.com 
@site: let me think 
@software: PyCharm 
@file: app.py   # #!/usr/bin/env python
@time: 18-4-9 下午4:41 
"""
import os, re
import pandas as pd
from flask import Flask, request, current_app, send_from_directory
import json, time, random
from elasticsearch import Elasticsearch

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
if not os.path.exists(file_dir):
    print("新建 文件夹：", file_dir)
    os.makedirs(file_dir)
ALLOWED_EXTENSIONS = set(['json', 'xls', 'xlsx', "xmind", "csv"])

es = Elasticsearch()


def es_query(query, index_name):
    """
    es 查询语句
    """
    s = es.search(index=index_name, body=query)
    s = [{"id": _["_source"]["qid"], "category": _["_source"]["category"],
          "title": _["_source"]["title"], "desc": _["_source"]["desc"],
          "answer": _["_source"]["answer"]} for _ in s['hits']['hits']]
    return s


# json
@app.route('/api/getResponse/', methods=['GET', 'POST'])
def getResponse():
    def form_or_json():
        data = request.get_json(silent=True)
        return data if data is not None else request.form

    data = form_or_json()
    print(data)
    # try:
    if request.method == "POST":
        data = request.data
        data = data.decode(encoding="utf-8")
        query = {
            "query": {
                "bool": {
                    "should": [
                        # {"terms": {
                        #     "category": ["烦恼"]}},
                        {"match": {"title": {"query": data, "boost": 3}}},
                        # "boost": 8，#
                    ]
                }
            }
        }
        a = es_query(query, "baike_qa")
        if a:
            result = {"status": 1, "result": a}
        else:
            result = {"status": 0, "info": "not supported!"}
        return json.dumps(result, indent=2, ensure_ascii=False)

    elif request.method == "GET":
        # caseId = request.args.get("caseId")
        result = {"status": 0, "info": "GET method is not allow!"}
        return json.dumps(result)

    else:
        result = {"status": 0, "info": "request method should be GET/POST ！"}
        return json.dumps(result)


# html
@app.route('/', methods=['GET', 'POST'])
def index():
    # 直接返回静态文件
    return app.send_static_file("dialogue.html")


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", "8001"))
    app.run(host='0.0.0.0', port=port, debug=True)

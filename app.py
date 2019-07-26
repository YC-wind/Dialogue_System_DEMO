#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-07-25 19:57
"""

import os, re
import pandas as pd
from flask import Flask, request, current_app, send_from_directory
import json, time, random
# from elasticsearch import Elasticsearch
import requests
from flask_cors import *

# app = Flask(__name__)


app = Flask(__name__)
CORS(app, supports_credentials=True, resources=r'/*')
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
if not os.path.exists(file_dir):
    print("新建 文件夹：", file_dir)
    os.makedirs(file_dir)
ALLOWED_EXTENSIONS = set(['json', 'xls', 'xlsx', "xmind", "csv"])


# es = Elasticsearch()


def get_response(data):
    """
    获取返回信息
    """
    if data["q"] == "0":
        return "hello"
    elif data["q"] == "image":
        return "请欣赏一下图片<br><img src='/static/robot/logo.png'/>"
    else:
        return "a nice day"


# json
@app.route('/api/getResponse/', methods=['GET', 'POST'])
def getResponse():
    # try:
    def form_or_json():
        data = request.get_json(silent=True)
        return data if data is not None else request.form

    data = form_or_json()
    print(data)
    if request.method == "POST":
        a = get_response(data)
        print(a)
        if a:
            result = {"status": 1, "answer": a}
        else:
            result = {"status": 0, "info": "not supported!"}
        return json.dumps(result, indent=2, ensure_ascii=False)

    elif request.method == "GET":
        q = request.args.get("q")
        a = get_response({"q": q})
        print(a)
        if a:
            result = {"status": 1, "answer": a, "suggestion": ["a", "b", "c"]}
        else:
            result = {"status": 0, "info": "not supported!"}
        callback = request.args.get("callback")
        # return json.dumps(result, indent=2, ensure_ascii=False)
        return callback + "(" + json.dumps(result, indent=2, ensure_ascii=False) + ")"

    else:
        result = {"status": 0, "info": "request method should be GET/POST ！"}
        return json.dumps(result)


# html
@app.route('/', methods=['GET', 'POST'])
def index():
    # 直接返回静态文件
    return app.send_static_file("index.html")


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", "8002"))
    app.run(host='127.0.0.1', port=port, debug=True)

#说明

这部分主要来源于 http://openkg.cn/tool/elasticsearch-kbqa

将Django 改为flask 轻量级web，同时将python2转为python3版本，处理编码问题

```
## 用elasticsearch搭建简易的语义搜索引擎

核心的代码在4个python文件中，其它的文件都是网站代码，或者数据文件，可以忽视

### preprocess.py
将实验的数据集转换为json格式，同时做一些预处理。处理后，一个实体及其所关联的所有属性和属性值存储为一个json对象，作为将导入elasticsearch的一个文档

### build_dict.py
构建数据集中属性名字典和实体名字典，用于在解析查询时判断查询语句是否包含知识库中的属性或实体。在数据集较小的时候可以这样做，在数据集较大的时候可以通过检索elasticsearch来判断

### insert.py
将处理好的数据导入elasticsearch, 需为elasticsearch新建index和type（这个没有写成代码，在命令行完成)

### search/views.py
核心代码，包括用户查询解析，elasticsearch查询构造。
```

为实验数据集新建index('demo')和type('person')。elasticsearch使用Restful API可以方便的交互，通过elasticsearch的mapping文件可以创建
index和type，并指定每个字段在elasticsearch中存储的类型。

下述示例用curl命令在命令行中与elasticsearch交互。其中, height, weight存储为integer数据类型，而实体名subj和其他属性存储为keyword类
型。所有其他属性存储在一个nested object对象中。 打开命令行，运行:

```bash
 curl -XPUT 'localhost:9200/demo?pretty' -H 'Content-Type: application/json' -d' 
 {
  "mappings": {
    "person": {
      "properties": {
        "subj": {
          "type": "keyword"
        },
        "height": {
          "type": "integer"
        },
        "weight": {
          "type": "integer"
        },
        "po": {
          "type": "nested",
          "properties": {
            "pred": {
              "type": "keyword"
            },
            "obj": {
              "type": "keyword"
            }
          }
        }
      }
    }
  }
}
 '
 
 
 {"mappings": {"person": {"properties": {"subj": {"type": "keyword"}, "height": {"type": "integer"}, "weight": {"type": "integer"},"po": {"type": "nested","properties": {"pred": {"type": "keyword"}, "obj": {"type": "keyword"}}}}}}}
 ```



与检索（多以文档进行展示）区分开！

## 环境

-  ES
-  python3.6
-  flask
-  ahocorasick

## 使用

- 下载安装ES
- ./elasticsearch -d -Xms512m -Xmx512m 后台运行
-  preprocess.py 预处理文件
-  insert.py 数据导入ES
-  python3 app.py 启动服务
-  http://127.0.0.1:8000/ 测试

## 测试
``` bash
 curl -XGET 'localhost:9200/demo/person/_search?&pretty' -H 'Content-Type:application/json' -d' {
"query":{ "bool":{
"must":{ "term":{"subj":"姚明"}
}}}}'

curl -XGET 'localhost:9200/analyze?&pretty' -H 'Content-Type:application/json' -d'{"analyzer":"standard","text":["我爱你"]}'

curl -XGET 'localhost:9200/demo/person/_search?&pretty' -H 'Content-Type:application/json' -d'{"query":{"bool":{"must":[{"range":{"height":{"gt":170}}}]}}}'
```
curl 结果
```json
{
  "took" : 1,
  "timed_out" : false,
  "_shards" : {
    "total" : 5,
    "successful" : 5,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : 1,
    "max_score" : 9.301947,
    "hits" : [
      {
        "_index" : "demo",
        "_type" : "person",
        "_id" : "19002",
        "_score" : 9.301947,
        "_source" : {
          "po" : [
            {
              "pred" : "alumniOf",
              "obj" : "上海体育技术教育学院"
            },
            {
              "pred" : "birthDate",
              "obj" : "1980年9月12日"
            },
            {
              "pred" : "birthPlace",
              "obj" : "上海"
            },
            {
              "pred" : "gender",
              "obj" : "男"
            },
            {
              "pred" : "nationality",
              "obj" : "中国"
            },
            {
              "pred" : "民族",
              "obj" : "汉族"
            },
            {
              "pred" : "职业",
              "obj" : "运动员"
            },
            {
              "pred" : "职业",
              "obj" : "篮球运动员"
            },
            {
              "pred" : "职业",
              "obj" : "其他"
            },
            {
              "pred" : "职业",
              "obj" : "上海大鲨鱼队老板"
            },
            {
              "pred" : "children",
              "obj" : "姚沁蕾"
            },
            {
              "pred" : "spouse",
              "obj" : "叶莉"
            }
          ],
          "height" : 226,
          "subj" : "姚明"
        }
      }
    ]
  }
}
```
姚明是谁

`http://127.0.0.1:8000/search/?question=%E5%A7%9A%E6%98%8E%E6%98%AF%E8%B0%81`

页面结果
```
属性名称	属性值
subj	姚明
height	226
alumniOf	上海体育技术教育学院
birthDate	1980年9月12日
birthPlace	上海
gender	男
nationality	中国
民族	汉族
职业	运动员 篮球运动员 其他 上海大鲨鱼队老板
children	姚沁蕾
spouse	叶莉
```

从结果来看，对 query的语义分析 处理这块，相对简单，很薄弱

这里只是将 query （处理简单，实体、属性抽取 会有错误，属性值条件筛选还需优化） 转化为 ES 查询API

## TODO

query 意图识别

句法分析，实体（槽）抽取

属性值条件查询

[ES数据库管理平台](https://www.elastic.co/cn/downloads/kibana)



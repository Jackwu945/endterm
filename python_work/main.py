import json
import random
import time

from flask import Flask, jsonify, make_response,render_template
from flask_restful import Api, Resource, reqparse
import pymysql
from educationgroupspider import *

usr_no = reqparse.RequestParser()
usr_no.add_argument("name", str)
usr_no.add_argument("pwd", str)

usr_info = reqparse.RequestParser()
usr_info.add_argument("stuno", str)
usr_info.add_argument("pwd", str)

app = Flask(__name__)
app.static_folder='static'
api = Api(app)
app.json.ensure_ascii = False # 解决中文乱码问题
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"


def querydb(table,no,db):


    cursor = db.cursor()
    cursor.execute("SELECT * FROM `{}` WHERE `no` = '{}'".format(table,no))

    data = cursor.fetchall()

    return data

def healthgrade(BMI,steps,sportstime):
    BMI = int(float(BMI))
    if BMI >25:
        score = 60
    elif BMI > 27:
        score =  30
    elif 18.5 <= BMI < 25:
        score = 100
    elif BMI < 18.5:
        score = 60
    else:
        score = 30

    score+= round(max(0,min(100,(int(steps)/10000)*100),1))
    score+= round(max(0,min(100,(int(sportstime)/60)*100),1))

    return round(score/3,2)

class englishelper(Resource):
    def get(self):
        try:
            with open('list.json', 'r', encoding='utf8') as fp:
                word_lst = json.load(fp)['list']
            word = random.choice(word_lst)
            res = make_response(jsonify(code=200, data=word.replace(" ","_")))
            # r = requests.get('https://cdn.jsdelivr.net/gh/lyc8503/baicizhan-word-meaning-API/data/words/{}.json'.format(word)).text
            # word_detword_detaill = json.loads(r)
            # mean=word_detword_detaill['mean_cn']
            # try:
            #     sentence=word_detword_detaill['sentence']
            #     sentence_cn=word_detword_detaill['sentence_trans']
            # except:
            #     sentence="No example provided."
            #     sentence_cn="无例句提供"
            res.status = '200'  # 设置状态码
            res.mimetype = 'application/json;charset=utf-8'  # 设置响应类型
            return res
        except Exception as e:
            res = make_response(
                jsonify(code=200, data=str(e)))
            res.status = '200'  # 设置状态码
            res.mimetype = 'application/json;charset=utf-8'  # 设置响应类型
            return res


class get_general(Resource):
    def post(self):
        db = pymysql.connect(host="localhost", user="mylife", password="sbchengxuke", db="mylife")
        args = usr_no.parse_args()

        activities = querydb('usr_activity',args['name'],db)
        todo = [t for t in querydb('usr_todo',args['name'],db) if t[-1] != 0]
        health = [querydb('usr_health_bmi',args['name'],db),querydb('usr_health_steps',args['name'],db),querydb('usr_health_heartbeat',args['name'],db),querydb('usr_health_sporttime',args['name'],db)]
        expend = querydb('usr_expend',args['name'],db)

        try:
            res = make_response(jsonify(code=200, data={'activities':activities,'todo':todo,'health':health,'expend':expend,'grade':healthgrade(health[0][0][2],health[1][0][2],health[3][0][2])}))
        except:
            res = make_response(jsonify(code=200, data={'activities':activities,'todo':todo,'health':health,'expend':expend,'grade':"无数据无法打分"}))

        res.status = '200'  # 设置状态码
        res.mimetype = 'application/json;charset=utf-8'  # 设置响应类型
        db.close()
        return res



class verify_usr(Resource):
    def post(self):
        db = pymysql.connect(host="localhost", user="mylife", password="sbchengxuke", db="mylife")
        args = usr_no.parse_args()
        no = querydb('usr_no',args['name'],db)
        db.close()

        if no != ():
            res = make_response(jsonify(code=200, data={"no":args['name'],'pwd':args['pwd']}))
            res.status = '200'  # 设置状态码
            res.mimetype = 'application/json;charset=utf-8'  # 设置响应类型
            time.sleep(2)
            return res
        else:
            res = make_response(jsonify(code=400, data='无此用户！'))
            res.status = '200'  # 设置状态码
            res.mimetype = 'application/json;charset=utf-8'  # 设置响应类型
            time.sleep(1)
            return res

class update_classtable(Resource):
    def post(self):
        args = usr_info.parse_args()
        sess, res = get_captcha()
        login(sess, res,no = args['stuno'],pwd = args['pwd'])
        r = get_class_table(sess).text

        # 或者使用json.dumps和make_response函数
        res = make_response(jsonify(code=200,data=r))
        res.status = '200'  # 设置状态码
        res.mimetype = 'application/json;charset=utf-8' # 设置响应类型
        return res

class dbtest(Resource):
    def get(self):
        args = usr_no.parse_args()
        r = querydb('usr_no',args['name'])

        res = make_response(jsonify(code=200, data=r))
        res.status = '200'  # 设置状态码
        res.mimetype = 'application/json;charset=utf-8'  # 设置响应类型

        return res

@app.route('/')
def mainsite():
    return render_template("usr_get.html")

@app.route('/mylife')
def mylife():
    return render_template("index.html")

@app.route('/res/bg')
def res():
    return render_template("/res/bg2.jpg")

api.add_resource(update_classtable,"/getclasstable")
api.add_resource(verify_usr,"/verify")
api.add_resource(get_general,"/general")
api.add_resource(dbtest,"/dbtest")
api.add_resource(englishelper,"/english")

if __name__ == '__main__':
    app.run(debug=True, host='10.226.8.13', port=8080)
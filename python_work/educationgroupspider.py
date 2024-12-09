import requests
import ddddocr
import base64

ocr=ddddocr.DdddOcr()

def getb64(no,pwd):
    b64no = str(base64.b64encode((no).encode('utf-8')))
    b64no=b64no[2:len(b64no)-1:]
    b64str="%%%"
    b64pwd = str(base64.b64encode((pwd).encode('utf-8')))
    b64pwd=b64pwd[2:len(b64pwd)-1:]

    return b64no+b64str+b64pwd

def get_class_unused(session):  # 废弃
    hearders={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}

    param = {'xnxqh': '2023-2024-1',
    'skyx': 'gW4AAABQjYbM567U','sknj': '2023','skzy': 'FE06C4040463F8CFE0530B98CE0A0CD1',
    'skbjid':'','skbj': '5','zc1': '1','zc2': '18','skxq1': '1','skxq2': '5','jc1': '01','jc2': '11'}

    r = requests.post('https://jw.educationgroup.cn/gzasc_jsxsd/kbcx/kbxx_xzb_ifr',headers=hearders,params=param,cookies=session.cookies)

    return r

def get_captcha(session):
    hearders={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}
    r=session.get('https://jw.educationgroup.cn/gzasc_jsxsd/verifycode.servlet?t=0.5262023160212345',headers=hearders)

    with open("gzasccaptcha.jpg",'wb')as f:
        f.write(r.content)

    code = ocr.classification(r.content)

    return session,code

def get_class_table(session):
    hearders={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}
    r = requests.post("https://jw.educationgroup.cn/gzasc_jsxsd/framework/main_index_loadkb.jsp",headers=hearders,cookies=session.cookies,params={'rq':'2023-12-18',"sjmsValue":"E37858B1799D43C9A598C8C6D1D21E05"})
    return r

def login(no,pwd):
    session=requests.session()
    session,code = get_captcha(session)
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}
    param={'userAccount': no,'userPassword': '','RANDOMCODE': code,'encoded':getb64(no,pwd)}
    # 发起请求
    response = session.post('https://jw.educationgroup.cn/gzasc_jsxsd/xk/LoginToXk',headers=headers,params=param)

    return session

    # # 获取重定向后的 url
    # if response.history:
    #     # 获取最后一个 Response 对象
    #     last_response = response.history[-1]
    #     location_url = last_response.headers['Location']
    #     print(location_url)

if __name__ == '__main__':
    sess = login('202310010388','Misspan250')

    r=get_class_table(sess)
    print(r.text)

    # with open ('classtable.html','w') as f:
    #     f.write(r.text)
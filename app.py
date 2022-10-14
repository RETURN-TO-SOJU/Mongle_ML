import os
from flask import Flask, request, jsonify
from model.kakao.katalk_parsing import katalk_msg_parse as parse
import json
import requests
import boto3
import unicodedata
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

UPLOAD_FOLDER = '/path/to/the/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

headers = {'Content-Type': 'application/json; charset=utf-8'}

@app.route('/')
def hello():
    return "ML server running"

@app.route('/katalk/<where>', methods=['POST', 'GET'])
def kakaoTalkService(where):
    s3c = boto3.client('s3')
    params = request.get_json()
    bucket = params['bucket']
    path = params['path']
    file_name = unicodedata.normalize('NFC', path.split('/')[-1])
    user_data = file_name.split('-')
    pk = user_data[0]
    nickname = user_data[1]
    room = user_data[2]
    s3c.download_file(bucket, path, os.path.join('./uploads', file_name))
    print("start: ", file_name)
    try:
        keyword, parsing, start, end = parse(os.path.join('./uploads', file_name), nickname)
        json_object = json.loads(parsing)
        data = {
            'user_pk': pk,
            'kakao_data': json_object,
            'start_date': start,
            'end_date': end,
            'keyword': keyword,
            'roomName': room
        }
        print("end ", file_name)
        os.remove(os.path.join('./uploads', file_name))
        if where == 'dev':
            res = requests.post('http://dev.mongle.org/api/ml/kakao', data=json.dumps(data), headers=headers)
            print('dev server serving')
        elif where == 'prod':
            res = requests.post('http://prod.mongle.org/api/ml/kakao', data=json.dumps(data), headers=headers)
            print('prod server serving')
        else:
            return jsonify({
                "answer": "요청 오류."
            })
    except requests.ConnectionError as e:
        print("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")
        print(str(e))
        pass
    except requests.Timeout as e:
        print("OOPS!! Timeout Error")
        print(str(e))
        pass
    except requests.RequestException as e:
        print("OOPS!! General Error")
        print(str(e))
        pass
    except KeyboardInterrupt:
        print("Someone closed the program")
        pass
    except:
        data = {
            'user_pk': pk
        }
        os.remove(os.path.join('./uploads', file_name))
        if where == 'dev':
            res = requests.post('http://dev.mongle.org/api/ml/error', data=json.dumps(data), headers=headers)
            print('dev server error serving')
        elif where == 'prod':
            res = requests.post('http://prod.mongle.org/api/ml/error', data=json.dumps(data), headers=headers)
            print('prod server error serving')

    return jsonify(
        {"message": str(res.status_code)}
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

import re
import pandas as pd
from util.emotion_class import LABELS
from model.emotion.classifier import trained_model as predict
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from model.AES.aes_256 import AESCipher


def katalk_msg_parse(file_path, nickname):
    my_katalk_data = list()
    # katalk_msg_pattern = "[0-9]{4}[년.] [0-9]{1,2}[일.] [0-9]{1,2}[오\S.] [0-9]{1,2}:[0-9]{1,2},.*:" #카카오톡 메시지 패턴(ios)
    katalk_msg_pattern = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2},.*:"  # 카카오톡 메시지 패턴(android)

    # date_info = "[0-9]{4}년 [0-9]{1,2}월 [0-9]{1,2}일 \S요일" #날짜 바뀌는 패턴(ios)
    date_info = "[0-9]{4}년 [0-9]{1,2}월 [0-9]{1,2}일 오\S [0-9]{1,2}:[0-9]{1,2}$"  # 날짜 바뀌는 패턴(android)

    in_out_info = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2}:.*"  # 나갔다는 알림 무시
    money_text = "[0-9]{4}[년.] [0-9]{1,2}[일.] [0-9]{1,2}[오\S.] [0-9]{1,2}:[0-9]{1,2},.*:.[0-9]{1,3},.*원"  # 카카오페이 돈거래
    audio_visual_text = "^동영상$|^사진$|^사진 [0-9]{1,2}장$|^이모티콘$|^봉투가 도착했어요.$|^삭제된 메시지입니다.&"  # 사진이나 동영상 메세지, 이모티콘은 추가 x
    with open(file_path) as f:
        password = f.readlines()[-1]
    global key
    key = password.strip()
    for line in open(file_path):
        if re.match(date_info, line) or re.match(in_out_info, line) or re.match(money_text, line):
            continue
        elif line == '\n':
            continue
        elif re.match(katalk_msg_pattern, line):
            line = line.split(",")  # ,기준 2020. 1. 23. 11:57, 윤승환 : 너 친구약속 만나?
            date_time = line[0]
            user_text = line[1].split(" : ", maxsplit=1)
            user_name = user_text[0].strip()
            text = user_text[1].strip()
            if re.match(audio_visual_text, text):  # 사진 짤
                continue
            elif my_katalk_data and my_katalk_data[-1]['user_name'] == user_name:  # 동일 인물의 발화에 대해서..
                my_katalk_data[-1]['text'] += " " + text
                my_katalk_data[-1]['len'] += len(text)
            else:
                my_katalk_data.append({"date_time": date_time,
                                       "user_name": user_name,
                                       "text": text,
                                       "len": len(text)})
        else:
            if len(my_katalk_data) > 0:
                my_katalk_data[-1]['text'] += "\n" + line.strip()  # 의도적으로 문장을 나눈 경우
    my_katalk_df = pd.DataFrame(my_katalk_data)
    data = customizing(my_katalk_df, nickname)
    start = re.findall(r'\d+', data['date_time'].iloc[0])
    start_date = start[0].zfill(4) + start[1].zfill(2) + start[2].zfill(2)
    end = re.findall(r'\d+', data['date_time'].iloc[-1])
    end_date = end[0].zfill(4) + end[1].zfill(2) + end[2].zfill(2)
    return make_keyword(data, start_date, end_date), \
           make_emotion(data), \
           start_date, \
           end_date


def customizing(data, nickname):
    print("20길이 이하 버리기")
    find_index = data[data['len'] < 20].index
    filtering = data.drop(find_index)
    my_n_index = filtering[filtering['user_name'] != nickname].index
    filtering = filtering.drop(my_n_index)
    print("문장길이 삭제 완료")
    return filtering.drop(['len'], axis=1)


def make_keyword(data, start_date, end_date):
    print("키워드 추출")
    okt = Okt()
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    dt_index = pd.date_range(start=start_date, end=end_date)
    dt_list = dt_index.strftime("%Y년 %-m월 %-d일").tolist()
    keyword = {}
    for i in dt_list:
        index = data['date_time'].apply(lambda x: x.startswith(i))
        text = ""
        for j in data[index]['text']:
            text += j + '\n'
        if not text:
            continue
        tokenized_doc = okt.pos(text)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
        n_gram_range = (1, 1)
        if len(tokenized_nouns) <= 1:
            continue
        count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
        candidates = count.get_feature_names_out()

        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)
        if len(candidates) < 5:
            keyword[i] = mmr(doc_embedding, candidate_embeddings, candidates, top_n=len(candidates), diversity=0.7)
        else:
            keyword[i] = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)
    print("키워드 추출 완료")
    return keyword

def make_emotion(data):
    global cnt
    cnt = 0
    global sum
    sum = len(data)
    print("감정분석 시작")
    data[['emotion']] = data.apply(add_emotion, axis=1)
    print("감정분석 완료")

    print("암호화 시작")
    data['text'] = data['text'].apply(text_encrypt)
    print("암호화 완료")
    return data.to_json(orient="records", indent=4)

#전역변수로 1/100 설정하기.
def add_emotion(data):
    global cnt, sum
    cnt += 1
    if (cnt % 10 == 0):
        print("감정분석 진행 = ", cnt / sum * 100)
    result = predict(data[2])[0]
    emotion = LABELS[result.max(dim=0).indices]
    return pd.Series([emotion])


def text_encrypt(data):
    result = AESCipher(bytes(key.encode("utf-8"))).encrypt(data)
    return pd.Series([result])


def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity = cosine_similarity(candidate_embeddings)
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


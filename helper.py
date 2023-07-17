# 사용되는 함수 모음 
import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer

def getIq(field):
    q1 = field.quantile(q=0.25)
    q3 = field.quantile(q=0.75)
    iqr = q3 - q1
    하한 = q1 - 1.5 * iqr
    상한 = q3 + 1.5 * iqr

    결측치경계 = [하한, 상한]
    return 결측치경계


def replaceOutlier(df, fieldName):
    cdf = df.copy()

    # fieldName이 List가 아니면 List로 변환 / 주어진 변수가 특정 데이터 타입인지 아닌지 (리스트인지아닌지) 검사한다.
    # 리스트 타입은 얕은 복사가 됨 (변수와 )
    if not isinstance(fieldName, list): 
        fieldName = [fieldName]

    for f in fieldName: # 
        결측치경계 = getIq(cdf[f])
        cdf.loc[cdf[f] < 결측치경계[0], f] = np.nan
        cdf.loc[cdf[f] > 결측치경계[1], f] = np.nan

    return cdf


# 결측치를 정제하는 함수 
def replaceMissingValue(df): # MisttingValue = 결측치
    imr = SimpleImputer(missing_values=np.nan, strategy='mean') 
    df_imr = imr.fit_transform(df.values) # fit으로 인공지능 모델에게 데이터를 학습시키고, transform으로 실제 적용하는 함수
    re_df = DataFrame(df_imr, index=df.index, columns=df.columns)
    return re_df # 정제한 데이터프레임을 리턴함
'''SimpleImputer : 각 열단위로 평균(strategy='mean')을 결측치(missing_values)에 지정 
strategy 옵션 : mean=평균, median=중앙값, most_frequent: 최빈값(가장 많이 관측되는 수)'''

def replaceMisttingValue(df) : 
    imr = SimpleImputer(missing_values=np.nan,strategy='mean')
    

def setCategory(df,ignore=[]) :
    cdf = df.copy()
    # 데이터 프레임의 변수명을 리스트로 변환
    ilist = list(cdf.dtypes.index)
    
    # 데이터 프레임의 변수형을 리스트로 변환
    vlist = list(cdf.dtypes.values)

    # 변수형에 대한 반복 처리
    for i, v in enumerate(vlist):
        # 변수형이 object이면?
        if v == 'object':
            # 변수명을 가져온다.
            field_name = ilist[i]
            # 가져온 변수명에 대해 값의 종류별로 빈도를 카운트 한 후 인덱스 이름순으로 정렬
            vc = cdf[field_name].value_counts().sort_index()
            #print(vc)

            # 인덱스 이름순으로 정렬된 값의 종류별로 반복 처리
            for ii, vv in enumerate(list(vc.index)):
                # 일련번호값 생성
                vnum = ii + 1
                #print(vv, " -->", vnum)

                # 일련번호값으로 치환
                cdf.loc[cdf[field_name] == vv, field_name] = vnum

            # 해당 변수의 데이터 타입을 범주형으로 변환
            cdf[field_name] = cdf[field_name].astype('category')    
    return cdf



# 워드클라우드 - 불용어 목록으로 쓰지 않는 단어 정제하기 
def clearStopwords(nouns, stopwords_file_path="wordcloud/stopwords-ko.txt"): # 불용어 목록이 default 값 
    with open(stopwords_file_path, 'r', encoding='utf-8') as f: # 불용어 목록을 읽어라 한줄씩 
        stopwords = f.readlines()
        
        for i, v in enumerate(stopwords): # 그런데 한줄 씩 읽으면 엔터가 먹히니까 엔터를 제외한다. 
            stopwords[i] = v.strip()

    data_set = [] 

    for v in nouns: # nlp.nouns를 사용해 명사들만 추출한 리스트를 불러온다. 
        if v not in stopwords: # 불용어에 없다면 
            data_set.append(v) # data_set에 추가한다. 

    return data_set # 정제된 명사 리스트를 반환한다. 

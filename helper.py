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
    






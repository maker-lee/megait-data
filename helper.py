"""
통계분석 유틸리티
@Author: 이광호(leekh4232@gmail.com)
"""
import numpy as np
from pandas import DataFrame, MultiIndex, concat, merge
from math import sqrt
from scipy.stats import t, pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro, normaltest, ks_2samp, bartlett, fligner, levene, chi2_contingency
from statsmodels.formula.api import ols
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from pca import pca


def getIq(field):
    """
    IQR(Interquartile Range)를 이용한 이상치 경계값 계산

    Parameters
    ------- 
    - field: 데이터 프레임의 필드

    Returns
    -------
    - 결측치경계: 이상치 경계값 리스트
    """
    q1 = field.quantile(q=0.25)
    q3 = field.quantile(q=0.75)
    iqr = q3 - q1
    하한 = q1 - 1.5 * iqr
    상한 = q3 + 1.5 * iqr
    결측치경계 = [하한, 상한]
    return 결측치경계


def replaceOutlier(df, fieldName):
    """
    이상치를 판별하여 결측치로 치환

    Parameters
    -------
    - df: 데이터 프레임
    - fieldName: 이상치를 판별할 필드명

    Returns
    -------
    - cdf : 결측치를 이상치로 치환한 데이터 프레임
    """
    cdf = df.copy()

    # fieldName이 List가 아니면 List로 변환
    if not isinstance(fieldName, list):
        fieldName = [fieldName]

    for f in fieldName:
        결측치경계 = getIq(cdf[f])
        cdf.loc[cdf[f] < 결측치경계[0], f] = np.nan
        cdf.loc[cdf[f] > 결측치경계[1], f] = np.nan

    return cdf


def replaceMissingValue(df, strategy='mean'):
    """
    결측치 정제

    Parameters
    -------
    - df: 데이터 프레임
    - strategy: 결측치 대체 전략(mean, median, most_frequent). 기본값은 mean

    Returns
    -------
    - re_df: 정제된 데이터 프레임
    """
    imr = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_imr = imr.fit_transform(df.values)
    re_df = DataFrame(df_imr, index=df.index, columns=df.columns)
    return re_df


def setCategory(df, fields=[]):
    """
    데이터 프레임에서 지정된 필드를 범주형으로 변경한다.

    Parameters
    -------
    - df: 데이터 프레임
    - fields: 범주형으로 변경할 필드명 리스트. 기본값은 빈 리스트(전체 필드 대상)

    Returns
    -------
    - cdf: 범주형으로 변경된 데이터 프레임
    """
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

            # 대상 필드 목록이 설정되지 않거나(전체필드 대상), 현재 필드가 대상 필드목록에 포함되어 있지 않다면?
            if not fields or field_name not in fields:
                continue

            # 가져온 변수명에 대해 값의 종류별로 빈도를 카운트 한 후 인덱스 이름순으로 정렬
            vc = cdf[field_name].value_counts().sort_index()
            # print(vc)

            # 인덱스 이름순으로 정렬된 값의 종류별로 반복 처리
            for ii, vv in enumerate(list(vc.index)):
                # 일련번호값 생성
                vnum = ii + 1
                # print(vv, " -->", vnum)

                # 일련번호값으로 치환
                cdf.loc[cdf[field_name] == vv, field_name] = vnum

            # 해당 변수의 데이터 타입을 범주형으로 변환
            cdf[field_name] = cdf[field_name].astype('category')

    return cdf


def clearStopwords(nouns, stopwords_file_path="wordcloud/stopwords-ko.txt"):
    """
    불용어를 제거한다.

    Parameters
    -------
    - nouns: 명사 리스트
    - stopwords_file_path: 불용어 파일 경로. 기본값은 wordcloud/stopwords-ko.txt

    Returns
    -------
    - data_set: 불용어가 제거된 명사 리스트
    """
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()

        for i, v in enumerate(stopwords):
            stopwords[i] = v.strip()

    data_set = []

    for v in nouns:
        if v not in stopwords:
            data_set.append(v)

    return data_set


def get_confidence_interval(data, clevel=0.95):
    """
    신뢰구간 계산

    Parameters
    -------
    - data: 데이터
    - clevel: 신뢰수준. 기본값은 0.95

    Returns
    -------
    - cmin: 신뢰구간 하한
    - cmax: 신뢰구간 상한
    """
    n = len(data)                           # 샘플 사이즈
    dof = n - 1                             # 자유도
    sample_mean = data.mean()               # 표본 평균
    sample_std = data.std(ddof=1)           # 표본 표준 편차
    sample_std_error = sample_std / sqrt(n)  # 표본 표준오차

    # 신뢰구간
    cmin, cmax = t.interval(
        clevel, dof, loc=sample_mean, scale=sample_std_error)

    return (cmin, cmax)


def normality_test(*any):
    """
    분산분석을 수행하기 위한 정규성을 검정 한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    names = []

    result = {
        'statistic': [],
        'p-value': [],
        'result': []
    }
    for i in any:
        s, p = shapiro(i)
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append(('정규성', 'shapiro', i.name))

    for i in any:
        s, p = normaltest(i)
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append(('정규성', 'normaltest', i.name))

    n = len(any)

    for i in range(0, n):
        j = i + 1 if i < n - 1 else 0

        s, p = ks_2samp(any[i], any[j])
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append(('정규성', 'ks_2samp', f'{any[i].name} vs {any[j].name}'))

    return DataFrame(result, index=MultiIndex.from_tuples(names, names=['condition', 'test', 'field']))


def equal_variance_test(*any):
    """
    분산분석을 수행하기 위한 등분산성을 검정 한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    s1, p1 = bartlett(*any)
    s2, p2 = fligner(*any)
    s3, p3 = levene(*any)

    names = []

    for i in any:
        names.append(i.name)

    fix = " vs "
    name = fix.join(names)
    index = [['등분산성', 'Bartlett', name], [
        '등분산성', 'Fligner', name], ['등분산성', 'Levene', name]]

    df = DataFrame({
        'statistic': [s1, s2, s3],
        'p-value': [p1, p2, p3],
        'result': [p1 > 0.05, p2 > 0.05, p3 > 0.05]
    }, index=MultiIndex.from_tuples(index, names=['condition', 'test', 'field']))

    return df


def independence_test(*any):
    """
    분산분석을 수행하기 위한 독립성을 검정한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    df = DataFrame(any).T
    result = chi2_contingency(df)

    names = []

    for i in any:
        names.append(i.name)

    fix = " vs "
    name = fix.join(names)

    index = [['독립성', 'Chi2', name]]

    df = DataFrame({
        'statistic': [result.statistic],
        'p-value': [result.pvalue],
        'result': [result.pvalue > 0.05]
    }, index=MultiIndex.from_tuples(index, names=['condition', 'test', 'field']))

    return df


def all_test(*any):
    """
    정규성, 등분산성, 독립성을 모두 검정한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    return concat([normality_test(*any), equal_variance_test(*any), independence_test(*any)])


def pearson_r(df):
    """
    피어슨 상관계수를 사용하여 상관분석을 수행한다.

    Parameters
    -------
    - df: 데이터 프레임

    Returns
    -------
    - rdf: 상관분석 결과 데이터 프레임
    """
    names = df.columns
    n = len(names)
    pv = 0.05

    data = []

    for i in range(0, n):
        # 기본적으로 i 다음 위치를 의미하지만 i가 마지막 인덱스일 경우 0으로 설정
        j = i + 1 if i < n - 1 else 0

        fields = names[i] + ' vs ' + names[j]
        s, p = pearsonr(df[names[i]], df[names[j]])
        result = p < pv

        data.append({'fields': fields, 'statistic': s,
                    'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)

    return rdf


def spearman_r(df):
    """
    스피어만 상관계수를 사용하여 상관분석을 수행한다.

    Parameters
    -------
    - df: 데이터 프레임

    Returns
    -------
    - rdf: 상관분석 결과 데이터 프레임
    """
    names = df.columns
    n = len(names)
    pv = 0.05

    data = []

    for i in range(0, n):
        # 기본적으로 i 다음 위치를 의미하지만 i가 마지막 인덱스일 경우 0으로 설정
        j = i + 1 if i < n - 1 else 0

        fields = names[i] + ' vs ' + names[j]
        s, p = spearmanr(df[names[i]], df[names[j]])
        result = p < pv

        data.append({'fields': fields, 'statistic': s,
                    'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)

    return rdf

def ext_ols(data, y, x):
    """
    회귀분석을 수해한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """

    # 독립변수의 이름이 리스트가 아니라면 리스트로 변환
    if type(x) != list:
        x = [x]

    # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
    expr = "%s~%s" % (y, "+".join(x))

    # 회귀모델 생성
    model = ols(expr, data=data)
    # 분석 수행
    fit = model.fit()

    # 파이썬 분석결과를 변수에 저장한다.
    summary = fit.summary()

    # 첫 번째, 세 번째 표의 내용을 딕셔너리로 분해
    my = {}

    for k in range(0, 3, 2):
        items = summary.tables[k].data
        # print(items)

        for item in items:
            # print(item)
            n = len(item)

            for i in range(0, n, 2):
                key = item[i].strip()[:-1]
                value = item[i+1].strip()

                if key and value:
                    my[key] = value

    # 두 번째 표의 내용을 딕셔너리로 분해하여 my에 추가
    my['variables'] = []
    name_list = list(data.columns)
    #print(name_list)

    for i, v in enumerate(summary.tables[1].data):
        if i == 0:
            continue

        # 변수의 이름
        name = v[0].strip()

        vif = 0

        # Intercept는 제외
        if name in name_list:
            # 변수의 이름 목록에서 현재 변수가 몇 번째 항목인지 찾기 
            j = name_list.index(name)
            vif = variance_inflation_factor(data, j)

        my['variables'].append({
            "name": name,
            "coef": v[1].strip(),
            "std err": v[2].strip(),
            "t": v[3].strip(),
            "P-value": v[4].strip(),
            "Beta": 0,
            "VIF": vif,
        })

    # 결과표를 데이터프레임으로 구성
    mylist = []
    yname_list = []
    xname_list = []

    for i in my['variables']:
        if i['name'] == 'Intercept':
            continue

        yname_list.append(y)
        xname_list.append(i['name'])

        item = {
            "B": i['coef'],
            "표준오차": i['std err'],
            "β": i['Beta'],
            "t": "%s*" % i['t'],
            "유의확률": i['P-value'],
            "VIF": i["VIF"]
        }

        mylist.append(item)

    table = DataFrame(mylist,
                   index=MultiIndex.from_arrays([yname_list, xname_list], names=['종속변수', '독립변수']))
    
    # 분석결과
    result = "𝑅(%s), 𝑅^2(%s), 𝐹(%s), 유의확률(%s), Durbin-Watson(%s)" % (my['R-squared'], my['Adj. R-squared'], my['F-statistic'], my['Prob (F-statistic)'], my['Durbin-Watson'])

    # 모형 적합도 보고
    goodness = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p < 0.05)." % (y, ",".join(x), "유의하다" if float(my['Prob (F-statistic)']) < 0.05 else "유의하지 않다", my['Df Model'], my['Df Residuals'], my['F-statistic'])

    # 독립변수 보고
    varstr = []

    for i, v in enumerate(my['variables']):
        if i == 0:
            continue
        
        s = "%s의 회귀계수는 %s(p%s0.05)로, %s에 대하여 %s."
        k = s % (v['name'], v['coef'], "<" if float(v['P-value']) < 0.05 else '>', y, '유의미한 예측변인인 것으로 나타났다' if float(v['P-value']) < 0.05 else '유의하지 않은 예측변인인 것으로 나타났다')

        varstr.append(k)

    # 리턴할
    return (model, fit, summary, table, result, goodness, varstr)



class OlsResult:
    def __init__(self):
        self._model = None
        self._fit = None
        self._summary = None
        self._table = None
        self._result = None
        self._goodness = None
        self._varstr = None

    @property
    def model(self):
        """
        분석모델
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        """
        분석결과 객체
        """
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        """
        분석결과 요약 보고
        """
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def table(self):
        """
        결과표
        """
        return self._table

    @table.setter
    def table(self, value):
        self._table = value

    @property
    def result(self):
        """
        결과표 부가 설명
        """
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def goodness(self):
        """
        모형 적합도 보고
        """
        return self._goodness

    @goodness.setter
    def goodness(self, value):
        self._goodness = value

    @property
    def varstr(self):
        """
        독립변수 보고
        """
        return self._varstr

    @varstr.setter
    def varstr(self, value):
        self._varstr = value


def my_ols(data, y, x):
    model, fit, summary, table, result, goodness, varstr = ext_ols(data, y, x)

    ols_result = OlsResult()
    ols_result.model = model
    ols_result.fit = fit
    ols_result.summary = summary
    ols_result.table = table
    ols_result.result = result
    ols_result.goodness = goodness
    ols_result.varstr = varstr

    return ols_result

def scalling(df, yname):
    """
    데이터 프레임을 표준화 한다.

    Parameters
    -------
    - df: 데이터 프레임
    - yname: 종속변수 이름

    Returns
    -------
    - x_train_std_df: 표준화된 독립변수 데이터 프레임
    - y_train_std_df: 표준화된 종속변수 데이터 프레임
    """
    x_train = df.drop([yname], axis=1)
    x_train_std = StandardScaler().fit_transform(x_train)
    x_train_std_df = DataFrame(x_train_std, columns=x_train.columns)
    
    y_train = df.filter([yname])
    y_train_std = StandardScaler().fit_transform(y_train)
    y_train_std_df = DataFrame(y_train_std, columns=y_train.columns)

    return (x_train_std_df, y_train_std_df)

def get_best_features(x_train_std_df):
    pca_model = pca()
    fit = pca_model.fit_transform(x_train_std_df)
    topfeat_df = fit['topfeat']
    
    best = topfeat_df.query("type=='best'")
    feature = list(set(list(best['feature'])))
    
    return (feature, topfeat_df)
import sys
import numpy as np

from math import sqrt

from pandas import DataFrame, MultiIndex, concat, DatetimeIndex
from scipy.stats import t, pearsonr, spearmanr, shapiro, normaltest, ks_2samp, bartlett, fligner, levene, chi2_contingency, norm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

from statsmodels.formula.api import ols, logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.gofplots import qqplot

from pca import pca

from matplotlib import pyplot as plt
import seaborn as sb

from tabulate import tabulate

from OlsResult import OlsResult
from LogitResult import LogitResult


class AnalysisHelper:
    """
    통계분석 유틸리티 클래스
    """
    def __init__(self):
        """
        생성자
        """
        plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
        plt.rcParams["font.size"] = 12
        plt.rcParams["figure.figsize"] = (15, 5)
        plt.rcParams["axes.unicode_minus"] = False

    def pretty_print(self, df, headers="keys", tablefmt="psql", numalign="right"):
        """
        데이터 프레임을 테이블 형식으로 출력

        Args
        -------
        - df (DataFrame): 데이터 프레임
        - headers (str or list, optional): 헤더 목록 Defaults to "keys".
        - tablefmt (str, optional): 테이블 형식. Defaults to "psql".
        - numalign (str, optional): 숫자 형식 정렬 방향. Defaults to "right".
        """
        print(tabulate(df, headers=headers, tablefmt=tablefmt, numalign=numalign))

    def set_datetime_index(self, df, field=None, inplace=False):
        """
        데이터 프레임의 인덱스를 datetime 형식으로 변환

        Parameters
        -------
        - df: 데이터 프레임
        - inplace: 원본 데이터 프레임에 적용 여부

        Returns
        -------
        - 인덱스가 datetime 형식으로 변환된 데이터 프레임
        """
        if inplace:
            if field is not None:
                df.set_index(field, inplace=True)

            df.index = DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
            df.sort_index(inplace=True)
        else:
            cdf = df.copy()

            if field is not None:
                cdf.set_index(field, inplace=True)

            cdf.index = DatetimeIndex(cdf.index.values, freq=cdf.index.inferred_freq)
            cdf.sort_index(inplace=True)
            return cdf

    def get_iq(self, field, is_print=True):
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
        극단치경계 = [하한, 상한]

        df = DataFrame({
            "극단치 경계": [하한, 상한]
        }, index=['하한', '상한'])

        if is_print:
            self.pretty_print(df)
        else:
            return 극단치경계

    def replace_outlier(self, df, fieldName):
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
            결측치경계 = self.get_iq(cdf[f], is_print=False)
            cdf.loc[cdf[f] < 결측치경계[0], f] = np.nan
            cdf.loc[cdf[f] > 결측치경계[1], f] = np.nan

        return cdf

    def replace_missing_value(self, df, strategy='mean'):
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

    def set_category(self, df, fields=[]):
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
                    # 일련번호값으로 치환
                    cdf.loc[cdf[field_name] == vv, field_name] = ii

                # 해당 변수의 데이터 타입을 범주형으로 변환
                cdf[field_name] = cdf[field_name].astype('category')

        return cdf

    def clear_stopwords(self, nouns, stopwords_file_path="wordcloud/stopwords-ko.txt"):
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

    def get_confidence_interval(self, data, clevel=0.95, is_print=True):
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

        if is_print:
            df = DataFrame({
                "신뢰구간": [cmin, cmax]
            }, index=['하한', '상한'])

            self.pretty_print(df)
        else:
            return (cmin, cmax)

    def normality_test(self, *any, is_print=True):
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
            'field': [],
            'test': [],
            'statistic': [],
            'p-value': [],
            'result': []
        }

        for i in any:
            s, p = shapiro(i)
            result['field'].append(i.name)
            result['test'].append('shapiro')
            result['statistic'].append(s)
            result['p-value'].append(p)
            result['result'].append(p > 0.05)
            names.append('정규성')

        for i in any:
            s, p = normaltest(i)
            result['field'].append(i.name)
            result['test'].append('shapiro')
            result['statistic'].append(s)
            result['p-value'].append(p)
            result['result'].append(p > 0.05)
            names.append('정규성')

        n = len(any)

        for i in range(0, n):
            j = i + 1 if i < n - 1 else 0

            s, p = ks_2samp(any[i], any[j])
            result['field'].append(f'{any[i].name} vs {any[j].name}')
            result['test'].append('ks_2samp')
            result['statistic'].append(s)
            result['p-value'].append(p)
            result['result'].append(p > 0.05)
            names.append('정규성')

        rdf = DataFrame(result, index=names)

        if is_print:
            self.pretty_print(rdf)
        else:
            return rdf

    def equal_variance_test(self, *any, is_print=True):
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
        index = ['등분산성', '등분산성', '등분산성']

        df = DataFrame({
            'field': [name, name, name],
            'test': ['Bartlett', 'Fligner', 'Levene'],
            'statistic': [s1, s2, s3],
            'p-value': [p1, p2, p3],
            'result': [p1 > 0.05, p2 > 0.05, p3 > 0.05]
        }, index=index)

        if is_print:
            self.pretty_print(df)
        else:
            return df

    def independence_test(self, *any, is_print=True):
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

        index = ['독립성']

        df = DataFrame({
            'field': [name],
            'test': ['Chi2'],
            'statistic': [result.statistic],
            'p-value': [result.pvalue],
            'result': [result.pvalue > 0.05]
        }, index=index)

        if is_print:
            self.pretty_print(df)
        else:
            return df

    def all_test(self, *any, is_print=True):
        """
        정규성, 등분산성, 독립성을 모두 검정한다.

        Parameters
        -------
        - any: 필드들

        Returns
        -------
        - df: 검정 결과 데이터 프레임
        """
        normality_test = self.normality_test(*any, is_print=False)
        equal_variance_test = self.equal_variance_test(*any, is_print=False)
        independence_test = self.independence_test(*any, is_print=False)
        
        cc = concat([normality_test, equal_variance_test, independence_test])

        if is_print:
            self.pretty_print(cc)
        else:
            return cc

    def pearson_r(self, df, is_print=True):
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

        if is_print:
            self.pretty_print(rdf)
        else:
            return rdf

    def spearman_r(self, df, is_print=True):
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

        if is_print:
            self.pretty_print(rdf)
        else:
            return rdf

    def scalling(self, df, yname=None):
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
        # 평소에는 yname을 제거한 항목을 사용
        # yname이 있지 않다면 df를 복사
        x_train = df.drop([yname], axis=1) if yname else df.copy()
        x_train_std = StandardScaler().fit_transform(x_train)
        x_train_std_df = DataFrame(x_train_std, columns=x_train.columns)

        if yname:
            y_train = df.filter([yname])
            y_train_std = StandardScaler().fit_transform(y_train)
            y_train_std_df = DataFrame(y_train_std, columns=y_train.columns)

        if yname:
            result = (x_train_std_df, y_train_std_df)
        else:
            result = x_train_std_df

        return result

    def get_best_features(self, x_train_std_df):
        pca_model = pca()
        fit = pca_model.fit_transform(x_train_std_df)
        topfeat_df = fit['topfeat']

        best = topfeat_df.query("type=='best'")
        feature = list(set(list(best['feature'])))

        return (feature, topfeat_df)

    def ols(self, data, y, x):
        """
        회귀분석을 수행한다.

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
        # print(name_list)

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
        result = "𝑅(%s), 𝑅^2(%s), 𝐹(%s), 유의확률(%s), Durbin-Watson(%s)" % (
            my['R-squared'], my['Adj. R-squared'], my['F-statistic'], my['Prob (F-statistic)'], my['Durbin-Watson'])

        # 모형 적합도 보고
        goodness = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p < 0.05)." % (y, ",".join(x), "유의하다" if float(
            my['Prob (F-statistic)']) < 0.05 else "유의하지 않다", my['Df Model'], my['Df Residuals'], my['F-statistic'])

        # 독립변수 보고
        varstr = []

        for i, v in enumerate(my['variables']):
            if i == 0:
                continue

            s = "%s의 회귀계수는 %s(p%s0.05)로, %s에 대하여 %s."
            k = s % (v['name'], v['coef'], "<" if float(v['P-value']) < 0.05 else '>', y,
                     '유의미한 예측변인인 것으로 나타났다' if float(v['P-value']) < 0.05 else '유의하지 않은 예측변인인 것으로 나타났다')

            varstr.append(k)

        ols_result = OlsResult()
        ols_result.model = model
        ols_result.fit = fit
        ols_result.summary = summary
        ols_result.table = table
        ols_result.result = result
        ols_result.goodness = goodness
        ols_result.varstr = varstr

        return ols_result

    def logit(self, data, y, x, subset=None):
        """
        로지스틱 회귀분석을 수행한다.

        Parameters
        -------
        - data : 데이터 프레임
        - y: 종속변수 이름
        - x: 독립변수의 이름들(리스트)
        """

        # 데이터프레임 복사
        df = data.copy()

        # 독립변수의 이름이 리스트가 아니라면 리스트로 변환
        if type(x) != list:
            x = [x]

        # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
        expr = "%s~%s" % (y, "+".join(x))

        # 회귀모델 생성
        model = logit(expr, data=df)
        # 분석 수행
        fit = model.fit()

        # 파이썬 분석결과를 변수에 저장한다.
        summary = fit.summary()

        # 의사결정계수
        prs = fit.prsquared

        # 예측결과를 데이터프레임에 추가
        df['예측값'] = fit.predict(df.drop([y], axis=1))
        df['예측결과'] = df['예측값'] > 0.5

        # 혼동행렬
        cm = confusion_matrix(df[y], df['예측결과'])
        tn, fp, fn, tp = cm.ravel()
        cmdf = DataFrame([[tn, tp], [fn, fp]], index=[
                         'True', 'False'], columns=['Negative', 'Positive'])

        # RAS
        ras = roc_auc_score(df[y], df['예측결과'])

        # 위양성율, 재현율, 임계값(사용안함)
        fpr, tpr, thresholds = roc_curve(df[y], df['예측결과'])

        # 정확도
        acc = accuracy_score(df[y], df['예측결과'])

        # 정밀도
        pre = precision_score(df[y], df['예측결과'])

        # 재현율
        recall = recall_score(df[y], df['예측결과'])

        # F1 score
        f1 = f1_score(df[y], df['예측결과'])

        # 위양성율
        fallout = fp / (fp + tn)

        # 특이성
        spe = 1 - fallout

        result_df = DataFrame({'설명력(Pseudo-Rsqe)': [fit.prsquared], '정확도(Accuracy)': [acc], '정밀도(Precision)': [pre], '재현율(Recall, TPR)': [
                              recall], '위양성율(Fallout, FPR)': [fallout], '특이성(Specificity, TNR)': [spe], 'RAS': [ras], 'f1_score': [f1]})

        # 오즈비
        coef = fit.params
        odds_rate = np.exp(coef)
        odds_rate_df = DataFrame(odds_rate, columns=['odds_rate'])

        logit_result = LogitResult()
        logit_result.model = model
        logit_result.fit = fit
        logit_result.summary = summary
        logit_result.prs = prs
        logit_result.cmdf = cmdf
        logit_result.result_df = result_df
        logit_result.odds_rate_df = odds_rate_df

        return logit_result

    def exp_time_data(self, data, yname, sd_model="m", max_diff=1):
        plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.unicode_minus"] = False

        df = data.copy()

        # 데이터 정상성 여부
        stationarity = False

        # 반복 수행 횟수
        count = 0

        # 결측치 존재 여부
        na_count = df[yname].isna().sum()
        print("결측치 수: %d" % na_count)

        plt.figure(figsize=(4, 5))
        sb.boxplot(data=df, y=yname)
        plt.show()
        plt.close()

        # 시계열 분해
        model_name = 'multiplicative' if sd_model == 'm' else 'additive'
        sd = seasonal_decompose(df[yname], model=model_name)

        figure = sd.plot()
        figure.set_figwidth(15)
        figure.set_figheight(16)
        fig, ax1, ax2, ax3, ax4 = figure.get_children()
        figure.subplots_adjust(hspace=0.4)

        ax1.set_ylabel("Original")
        ax1.grid(True)
        ax1.title.set_text("Original")
        ax2.grid(True)
        ax2.title.set_text("Trend")
        ax3.grid(True)
        ax3.title.set_text("Seasonal")
        ax4.grid(True)
        ax4.title.set_text("Residual")

        plt.show()

        # ACF, PACF 검정
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.subplots_adjust(hspace=0.4)

        sb.lineplot(data=df, x=df.index, y=yname, ax=ax1)
        ax1.title.set_text("Original")

        plot_acf(df[yname], ax=ax2)
        ax2.title.set_text("ACF Test")

        plot_pacf(df[yname], ax=ax3)
        ax3.title.set_text("PACF Test")

        plt.show()
        plt.close()

        while not stationarity:
            if count == 0:
                print("=========== 원본 데이터 ===========")
            else:
                print("=========== %d차 차분 데이터 ===========" % count)

            # ADF Test
            ar = adfuller(df[yname])

            ardict = {
                '검정통계량(ADF Statistic)': [ar[0]],
                '유의수준(p-value)': [ar[1]],
                '최적차수(num of lags)': [ar[2]],
                '관측치 개수(num of observations)': [ar[3]]
            }

            for key, value in ar[4].items():
                ardict['기각값(Critical Values) %s' % key] = value

            stationarity = ar[1] < 0.05
            ardict['데이터 정상성 여부(0=Flase,1=True)'] = stationarity

            ardf = DataFrame(ardict, index=['ADF Test']).T

            print(tabulate(ardf, headers=["ADF Test", ""], tablefmt='psql', numalign="right"))

            # 차분 수행
            df = df.diff().dropna()

            # 반복을 계속할지 여부 판단
            count += 1
            if count == max_diff:
                break

    def arima_diagnostics(self, resids, n_lags=40):

        # create placeholder subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        r = resids
        resids = (r - np.nanmean(r)) / np.nanstd(r)
        resids_nonmissing = resids[~(np.isnan(resids))]

        # residuals over time
        sb.lineplot(x=np.arange(len(resids)), y=resids, ax=ax1)
        ax1.set_title('Standardized residuals')
        ax1.grid()
        ax1.set_xlim(ax1.get_xlim())
        sb.lineplot(x=ax1.get_xlim(), y=[0, 0], linestyle='--', color='red', alpha=0.7, ax=ax1)

        # distribution of residuals
        x_lim = (-1.96 * 2, 1.96 * 2)
        r_range = np.linspace(x_lim[0], x_lim[1])
        norm_pdf = norm.pdf(r_range)

        sb.histplot(resids_nonmissing, kde=True, ax=ax2)
        ax2.plot(r_range, norm_pdf, 'g', lw=2, label='N(0,1)')
        ax2.set_title('Distribution of standardized residuals')
        ax2.set_xlim(x_lim)
        ax2.legend()
        ax2.grid()

        # Q-Q plot
        qq = qqplot(resids_nonmissing, line='s', ax=ax3)
        ax3.set_title('Q-Q plot')
        ax3.grid()

        # ACF plot
        plot_acf(resids, ax=ax4, alpha=0.05)
        ax4.set_title('ACF plot')
        ax4.grid()

        plt.show()
        plt.close()


helper = AnalysisHelper()

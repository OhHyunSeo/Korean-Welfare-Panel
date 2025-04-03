#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:40:41 2025

@author: oh
pdf 제출 13일 17시 40분
제출 메일 주소 : aiffall@naver.com
주제
    - 하위 목표 3개 이상
    - 각 하위목표에 대한 시각화 및 결론
    - 각 하위목표 결론을 토대로 전체 주제에 대한 결론
    
주제 : 직업군별 재산의 평등도에 대해 어떻게 생각할까? 
    하위 주제 1. 한국사회의 소득격차에 대한 사람들의 인식 / 소득격차가 매우 큰가? - wc14_5aq1
    하위 주제 2. 현재 한국사회의 소득 재산의 평등 정도? - wc14_3
    하위 주제 3. 소득격차 해소가 정부의 책임이라고 생각하는 사람들의 분포 - wc14_4
"""
# 주제 - 


# 1. 패키지 설치 및 모듈
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정
mpl.rc('font', family='AppleGothic')  # 맥 기본 한글 폰트
mpl.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# 2. 데이터 불러오기
raw_welfare = pd.read_spss("./data/Koweps_hpwc14_2019_beta2.sav")

# 3. 데이터 복사본 만들기
welfare = raw_welfare.copy()

welfare.shape

# 4. 변수 이름 변경
welfare = welfare.rename(columns = {'h14_g3' : 'sex',   # 성별
                                    'h14_g4' : 'birth', # 출생년도
                                    'h14_g10' : 'marriage_type', # 혼인 상태
                                    'h14_g11' : 'religion', # 종교
                                    'p1402_8aq1' : 'income', # 월급
                                    'h14_eco9' : 'code_job', # 직업 코드
                                    'h14_reg7' : 'code_region', # 지역 코드
                                    'wc14_5aq1' : 'income_diff', # 소득 격차
                                    'wc14_3': 'equal_wealth', # 재산 평등
                                    'wc14_4': 'gov_diff'}) # 정부 책임

print(welfare['job_category'].dtype)
print(welfare['code_job'].dtype)


# job_category를 맨 앞자리 숫자로 설정
welfare['job_category'] = welfare['code_job'].astype(str).str.split('.').str[0].str.zfill(4).str[:2]

# 맨 앞자리 숫자에 대한 대분류 매핑
job_category_map = {
    '01': '관리자급',
    '02': '전문가 및 관련 종사자',
    '03': '사무 종사자',
    '04': '서비스 종사자',
    '05': '판매 종사자',
    '06': '농림어업 종사자',
    '07': '기능원 및 장치 조작원',
    '08': '기계 및 조립 종사자',
    '09': '단순 노동 종사자'
    # '10': '군인'
}

# job_category_name 매핑
welfare['job_category_name'] = welfare['job_category'].map(job_category_map)

# 결과 확인
welfare[['code_job', 'job_category', 'job_category_name']].head()
# 직군별 인원 확인
welfare['job_category_name'].value_counts()
'''
job_category_name
단순 노동 종사자       1432
농림어업 종사자        1029
전문가 및 관련 종사자     962
사무 종사자           895
서비스 종사자          687
기계 및 조립 종사자      573
판매 종사자           569
기능원 및 장치 조작원     516
관리자              194
군인                21
Name: count, dtype: int64
'''

#-------------------------------------------------
# 1. 한국사회의 소득격차에 대한 사람들의 인식 / 소득격차가 매우 큰가?
'''
"1. 매우 동의한다       2. 동의한다          3. 동의도 반대도 하지 않는다
4. 반대한다              5. 매우 반대한다   6. 선택할 수 없음"
'''
welfare['income_diff'].dtypes
welfare['income_diff'].value_counts(dropna = False)
'''
wc14_5aq1
2.0    1170 # 동의
1.0     626 # 매우 동의
3.0     186 # 중립
4.0      26 # 반대
5.0      10 # 매우 반대
6.0       9 # 선택X
Name: count, dtype: int64
'''
# 응답 변수 이름 변경
response_map = {
    1.0: '매우 동의',
    2.0: '동의',
    3.0: '중립',
    4.0: '반대',
    5.0: '매우 반대',
    6.0: '선택할 수 없다'
}
welfare['wc14_5aq1_label'] = welfare['income_diff'].replace(response_map)

# welfare['wc14_5aq1_label'] = welfare['wc14_5aq1'].replace(response_map)

# 직업군별 동의 사항
welfare.groupby('job_category_name')['income_diff'].value_counts(normalize=True)
'''
job_category_name  wc14_5aq1
기계 및 조립 종사자        2.0          0.570370
                   1.0          0.362963
                   3.0          0.051852
                   6.0          0.007407
                   4.0          0.007407
기능원 및 장치 조작원       2.0          0.509091
                   1.0          0.381818
                   3.0          0.090909
                   4.0          0.018182
농림어업 종사자           2.0          0.641509
                   1.0          0.235849
                   3.0          0.084906
                   5.0          0.018868
                   4.0          0.009434
                   6.0          0.009434
단순 노동 종사자          2.0          0.516949
                   1.0          0.364407
                   3.0          0.080508
                   4.0          0.016949
                   6.0          0.012712
                   5.0          0.008475
사무 종사자             2.0          0.605505
                   1.0          0.279817
                   3.0          0.110092
                   5.0          0.004587
서비스 종사자            2.0          0.548611
                   1.0          0.326389
                   3.0          0.111111
                   4.0          0.013889
전문가 및 관련 종사자       2.0          0.616788
                   1.0          0.266423
                   3.0          0.102190
                   4.0          0.010949
                   5.0          0.003650
판매 종사자             2.0          0.569231
                   1.0          0.323077
                   3.0          0.084615
                   4.0          0.015385
                   6.0          0.007692
Name: proportion, dtype: float64
'''
# 데이터 변환: 직업군별 wc14_5aq1 비율을 피벗 테이블로 변환
wc14_pivot = welfare.groupby('job_category_name')['wc14_5aq1_label'].value_counts(normalize=True).unstack()

# 그래프 그리기
wc14_pivot.plot(kind='bar', figsize=(12, 6), cmap='tab10')

plt.title('직업군별 "소득격차가 큰가?"에 대한 응답 비율', fontsize=14)
plt.xlabel('직업군', fontsize=12)
plt.ylabel('비율', fontsize=12)
# y축을 백분율로 변환
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
plt.legend(title='응답 값',loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.show()

#-------------------------------------------------

# 2. 현재 한국사회의 소득 재산의 평등 정도? - wc14_3
'''
1. 매우 평등하다 <------------------> 7. 매우 불평등하다
'''
welfare['equal_wealth'].dtypes
welfare['equal_wealth'].value_counts()
'''
wc14_3
5.0    659
6.0    524
4.0    408
7.0    223
3.0    145
2.0     49
1.0     19
Name: count, dtype: int64
'''
welfare['equal_wealth'].isna().sum()

# 응답 변수 변경
response_map = {
    1.0: '매우 평등',
    2.0: '조금 평등',
    3.0: '평등',
    4.0: '중립',
    5.0: '불평등',
    6.0: '조금 불평등',
    7.0: '매우 불평등'
}
welfare['equal_wealth_label'] = welfare['equal_wealth'].replace(response_map)

# 직업군별 응답 현황
welfare.groupby('job_category_name')['equal_wealth'].value_counts(normalize=True)
'''
job_category_name  equal_income
기계 및 조립 종사자        5.0             0.370370
                   6.0             0.296296
                   4.0             0.133333
                   7.0             0.118519
                   3.0             0.074074
                   1.0             0.007407
기능원 및 장치 조작원       5.0             0.300000
                   6.0             0.254545
                   4.0             0.245455
                   7.0             0.136364
                   3.0             0.027273
                   2.0             0.027273
                   1.0             0.009091
농림어업 종사자           5.0             0.320755
                   6.0             0.245283
                   4.0             0.216981
                   7.0             0.141509
                   3.0             0.047170
                   2.0             0.028302
단순 노동 종사자          5.0             0.292373
                   6.0             0.288136
                   4.0             0.203390
                   7.0             0.097458
                   3.0             0.076271
                   2.0             0.021186
                   1.0             0.021186
사무 종사자             5.0             0.376147
                   6.0             0.270642
                   4.0             0.206422
                   7.0             0.073394
                   3.0             0.041284
                   2.0             0.022936
                   1.0             0.009174
서비스 종사자            4.0             0.284722
                   6.0             0.256944
                   5.0             0.236111
                   7.0             0.125000
                   3.0             0.062500
                   2.0             0.027778
                   1.0             0.006944
전문가 및 관련 종사자       5.0             0.379562
                   6.0             0.259124
                   4.0             0.167883
                   3.0             0.091241
                   7.0             0.069343
                   2.0             0.032847
판매 종사자             5.0             0.261538
                   6.0             0.230769
                   4.0             0.184615
                   7.0             0.176923
                   3.0             0.100000
                   2.0             0.046154
'''
# 데이터 변환: 직업군별 equal_wealth 비율을 피벗 테이블로 변환
equal_wealth_pivot = welfare.groupby('job_category_name')['equal_wealth_label'].value_counts(normalize=True).unstack()

equal_wealth_pivot.plot(kind='bar', figsize=(12, 6), cmap='tab10')

plt.title('직업군별 "재산의 평등성"에 대한 응답 비율', fontsize=14)
plt.xlabel('직업군', fontsize=12)
plt.ylabel('비율', fontsize=12)
# y축을 백분율로 변환
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
plt.legend(title='응답 값',loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.show()

#-------------------------------------------------

# 3. 소득격차 해소가 정부의 책임이라고 생각하는 사람들의 분포 - wc14_4
welfare['gov_diff'].dtypes
welfare['gov_diff'].value_counts()
'''
wc14_4
2.0    1016
3.0     458
1.0     321
4.0     207
5.0      13
6.0      12
Name: count, dtype: int64
'''
# 응답 변수 변경
response_map = {
    1.0: '매우 동의',
    2.0: '동의',
    3.0: '중립',
    4.0: '반대',
    5.0: '매우 반대',
    6.0: '선택할 수 없다'
}
welfare['gov_diff_label'] = welfare['gov_diff'].replace(response_map)

# 직업군별 응답 현황
welfare.groupby('job_category_name')['gov_diff'].value_counts(normalize=True)
'''
job_category_name  gov_diff
기계 및 조립 종사자        2.0         0.474074
                   1.0         0.266667
                   3.0         0.162963
                   4.0         0.081481
                   5.0         0.014815
기능원 및 장치 조작원       2.0         0.536364
                   3.0         0.227273
                   1.0         0.136364
                   4.0         0.100000
농림어업 종사자           2.0         0.556604
                   3.0         0.169811
                   1.0         0.160377
                   4.0         0.113208
단순 노동 종사자          2.0         0.474576
                   3.0         0.207627
                   1.0         0.199153
                   4.0         0.097458
                   6.0         0.021186
사무 종사자             2.0         0.458716
                   3.0         0.298165
                   4.0         0.114679
                   1.0         0.114679
                   5.0         0.009174
                   6.0         0.004587
서비스 종사자            2.0         0.493056
                   3.0         0.250000
                   1.0         0.138889
                   4.0         0.111111
                   6.0         0.006944
전문가 및 관련 종사자       2.0         0.510949
                   3.0         0.240876
                   1.0         0.131387
                   4.0         0.098540
                   5.0         0.014599
                   6.0         0.003650
판매 종사자             2.0         0.500000
                   3.0         0.223077
                   1.0         0.192308
                   4.0         0.069231
                   5.0         0.007692
                   6.0         0.007692
Name: proportion, dtype: float64
'''
# 데이터 변환: 직업군별 gov_diff 비율을 피벗 테이블로 변환
gov_diff_pivot = welfare.groupby('job_category_name')['gov_diff_label'].value_counts(normalize=True).unstack()

gov_diff_pivot.plot(kind='bar', figsize=(12, 6), cmap='tab10')

plt.title('직업군별 "소득격차 해소는 정부의 책임인가?"에 대한 응답 비율', fontsize=14)
plt.xlabel('직업군', fontsize=12)
plt.ylabel('비율', fontsize=12)
# y축을 백분율로 변환
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
plt.legend(title='응답 값',loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.show()





















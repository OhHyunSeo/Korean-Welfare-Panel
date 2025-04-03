#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:08:54 2025

@author: oh
korean.life.py
한국복지패널 데이터
"""
'''
데이터 분석 절차
1. 변수(컬럼) 검토 및 전처리
    변수의 특징 파악 -> 이상치, 결측치 정제 -> 분석의 용이성(변수의 값처리)
    전처리 : 분석할 변수를 각각 진행
2. 변수간의 관계 분석
    2-1. 요약 테이블
    2-2. 시각화
'''

# 1. 패키지 설치 및 모듈
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. 데이터 불러오기
raw_welfare = pd.read_spss("./data/Koweps_hpwc14_2019_beta2.sav")

# 3. 복사본을 이용해서 데이터 검토
welfare = raw_welfare.copy()

# 3.1 위와 아래 확인
welfare.head()
welfare.tail()
# 3.2 행, 열수 확인
welfare.shape # (14418, 830)
# 3.3 변수속성
welfare.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14418 entries, 0 to 14417
Columns: 830 entries, h14_id to h14_pers_income5
dtypes: float64(826), object(4)
memory usage: 91.3+ MB
'''

# 3.4 요약 통계
welfare.describe()
'''
h14_id       h14_ind  ...  h14_pers_income4  h14_pers_income5
count  14418.000000  14418.000000  ...      14418.000000        715.000000
mean    4672.108406      3.121723  ...          2.038702       1183.292308
std     2792.998128      3.297963  ...         32.965477       2147.418274
min        2.000000      1.000000  ...          0.000000     -10600.000000
25%     2356.000000      1.000000  ...          0.000000        206.000000
50%     4535.000000      1.000000  ...          0.000000        530.000000
75%     6616.000000      7.000000  ...          0.000000       1295.000000
max     9800.000000     14.000000  ...       3000.000000      22644.000000

[8 rows x 826 columns]
'''

# 성별, 태어난 연도, 혼인 상태, 종교, 월급, 직업 코드, 지역 코드

# 4. 변수명 변경

welfare = welfare.rename(columns = {'h14_g3' : 'sex',   # 성병
                                    'h14_g4' : 'birth', # 출생년도
                                    'h14_g10' : 'marriage_type', # 혼인 상태
                                    'h14_g11' : 'religion', # 종교
                                    'p1402_8aq1' : 'income', # 월급
                                    'h14_eco9' : 'code_job', # 직업 코드
                                    'h14_reg7' : 'code_region'}) # 지역 코드

'''
데이터 분석 절차
1. 변수(컬럼) 검토 및 전처리
    변수의 특징 파악 -> 이상치, 결측치 정제 -> 분석의 용이성(변수의 값처리)
    전처리 : 분석할 변수를 각각 진행
2. 변수간의 관계 분석
    2-1. 요약 테이블
    2-2. 시각화
'''
'''
주제 : 한국사람의 삶의 질
하위 목표 : 1. 성별에 따른 월급차이
'''
# 성별 변수 검토 및 전처리하기
# 1. 변수 검토 : 타입 파악, 범주마다 데이터의 형식
welfare['sex'].dtypes   # dtype('float64')

# 2. 빈도 구하기 (이상치 확인 가능)
welfare['sex'].value_counts()
'''
sex
2.0    7913
1.0    6505
Name: count, dtype: int64
'''

# 3. 이상치 확인
welfare['sex'].value_counts()

# 만약 이상치 발견시, 
# 이상치를 결측처리하고 : np.where()
welfare['sex'] = np.where(welfare['sex'] == 9, np.nan, welfare['sex'])
# 결측치를 확인하고 : isna().sum()
welfare['sex'].isna().sum()
# 결측치를 제거하고 : dropna()

# 성별 항목 이름 부여
welfare['sex'] = np.where(welfare['sex'] == 1, 'male', 'female')

welfare['sex'].value_counts()
'''
sex
female    7913
male      6505
Name: count, dtype: int64
'''
# 빈도 막대 시각화
sns.countplot(data = welfare, x = 'sex')
plt.show()

# 월급 변수 컴토 및 전처리
welfare['income'].dtypes # dtype('float64')

welfare['income'].describe()
'''
count    4534.000000
mean      268.455007
std       198.021206
min         0.000000
25%       150.000000
50%       220.000000
75%       345.750000
max      1892.000000
Name: income, dtype: float64
'''
# 히스토그램

sns.histplot(data = welfare, x = 'income')
plt.show()


# 이상치 확인
welfare['income'].describe()
'''
count    4534.000000
mean      268.455007
std       198.021206
min         0.000000
25%       150.000000
50%       220.000000
75%       345.750000
max      1892.000000
Name: income, dtype: float64
'''
# 결측치 확인
welfare['income'].isna().sum() # 9884

# 성별에 따른 월급 차이 분석
# 성별 월급 평균포
# income 결측치 제처
# sex별 분리
# income 평균 구하기
sex_income = welfare.dropna(subset = 'income').groupby('sex', as_index = False).agg(mean_income = ('income', 'mean'))
'''
      sex  mean_income
0  female   186.293096
1    male   349.037571
'''

sns.barplot(data = sex_income, x = 'sex', y = 'mean_income')
plt.show()

'''
나이와 월급의 관계 - 몇 살 때 월급을 가장 많이 받을까?
'''

welfare['birth'].describe()
'''
count    14418.000000
mean      1969.280205
std         24.402250
min       1907.000000
25%       1948.000000
50%       1968.000000
75%       1990.000000
max       2018.000000
Name: birth, dtype: float64
'''

sns.histplot(data = welfare, x = 'birth')
plt.show()

welfare['birth'].isna().sum() # 0

# 파생변수 만들기 - 나
welfare = welfare.assign(age = 2019 - welfare['birth'] + 1)
welfare['age'].describe()
'''
count    14418.000000
mean        50.719795
std         24.402250
min          2.000000
25%         30.000000
50%         52.000000
75%         72.000000
max        113.000000
Name: age, dtype: float64
'''

sns.histplot(data = welfare, x = 'age')
plt.show()


## 나이별 월급 평균표
# income 결측치 제거
# age 별 분리
# income 평균 구하기

age_income = welfare.dropna(subset = 'income').groupby('age').agg(mean_income = ('income', 'mean'))
'''
      mean_income
age              
19.0   162.000000
20.0   121.333333
21.0   136.400000
22.0   123.666667
23.0   179.676471
          ...
88.0    27.000000
89.0    27.000000
90.0    27.000000
91.0    20.000000
92.0    27.000000

[74 rows x 1 columns]
'''

# 선그래프
sns.lineplot(data = age_income, x = 'age', y = 'mean_income')
plt.show()

'''
20 대 초반 : 150만원
40~ 50대 : 최고점
60대 이후 : 감소추세
'''
'''
연령대에 따른 월급 차이
1. 초반(초년생, 30세 미만)
2. 중반(중년)
3. 후반(노년, 59세 이상)
'''

welfare['age'].head()
'''
0    75.0
1    72.0
2    78.0
3    58.0
4    57.0
Name: age, dtype: float64
'''
# 연령대 변수 만들기
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young',
                                         np.where(welfare['age'] <= 59, 'middle','old')))

# 빈도 구하기
welfare['ageg'].value_counts()
'''
ageg
old       5955
middle    4963
young     3500
Name: count, dtype: int64
'''

sns.countplot(data = welfare, x = 'ageg')

# 연령대별 월급 평균표
# income 결측치 제거
# ageg별 분리
# income 평균구하기
ageg_income = welfare.dropna(subset = 'income').groupby('ageg', as_index = False).agg(mean_income = ('income','mean'))
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income')

ageg_income = welfare.dropna(subset = 'income').groupby('ageg', as_index = False).agg(mean_income = ('income','mean'))
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income', order = ['young','middle','old'])
'''
중년 330만원
노년 140만원
초년 195만원
'''
'''
연령대 및 성별 월급 차이 - 성별 월급 차이는 연령대별로 다를까?
'''

# 연령대 및 성별 평균표
# income 결측치 제거
# ageg 및 sex 별 분리
# income 평균 구하기
sex_income = welfare.dropna(subset = 'income').groupby(['ageg','sex'], as_index = False).agg(mean_income = ('income', 'mean'))
'''
     ageg     sex  mean_income
0  middle  female   230.481735
1  middle    male   409.541228
2     old  female    90.228896
3     old    male   204.570231
4   young  female   189.822222
5   young    male   204.909548
'''

sns.barplot(data = sex_income, x = 'ageg', y = 'mean_income', hue = 'sex', order = ['young','middle','old'])
'''
초년 : 남녀 급여차이가 크지 않다
중년, 노년 : 남녀 급여 차이가 발생
'''
'''
나이 및 성별 월급 차이
'''
# 나이 및 성별 월급 평균표
# income 결측치 제거
# ageg 및 sex 별 분리
# income 평균 구하기

sex_age = welfare.dropna(subset = 'income').groupby(['age','sex'], as_index = False).agg(mean_income = ('income', 'mean'))
sex_age.head()
'''
    age     sex  mean_income
0  19.0    male   162.000000
1  20.0  female    87.666667
2  20.0    male   155.000000
3  21.0  female   124.000000
4  21.0    male   186.000000
'''
# 선그래프
sns.lineplot(data = sex_age, x = 'age', y = 'mean_income', hue = 'sex')

'''
남자 : 50세를 기준으로 증가, 이후로는 급격히 감소
여자 : 30세를 기준으로 증가, 이후로는 완만하게 감
성별 급여 차이 : 30세 이후로 차이가 급격히 벌어짐
'''
'''
직업별 월급 차이  - 어떤 직업이 월급을 가장 많이 받을까?
'''

welfare['code_job'].dtypes

welfare['code_job'].value_counts()
'''
code_job
611.0    962
941.0    391
521.0    354
312.0    275
873.0    236

112.0      2
784.0      2
423.0      1
861.0      1
872.0      1
Name: count, Length: 150, dtype: int64
'''

# 전처리하기
list_job = pd.read_excel("./data/Koweps_Codebook_2019.xlsx",
                       sheet_name = '직종코드')
list_job.head()
'''
   code_job                     job
0       111  의회 의원∙고위 공무원 및 공공단체 임원
1       112                기업 고위 임원
2       121          행정 및 경영 지원 관리자
3       122         마케팅 및 광고∙홍보 관리자
4       131       연구∙교육 및 법률 관련 관리자
'''
list_job.shape
# (156, 2)

# welfare에 list_job 결합

welfare = welfare.merge(list_job, how = 'left', on = 'code_job')

# code_job 에 결측치 제거하고 code_job, job 출력
welfare.dropna(subset = 'code_job')[['code_job','job']].head()
'''
    code_job               job
2      762.0               전기공
3      855.0       금속기계 부품 조립원
7      941.0       청소원 및 환경미화원
8      999.0  기타 서비스 관련 단순 종사자
14     312.0         경영 관련 사무원
'''

# 직업별 월급 평균표
# job, income 결측치 제거
# job별 분리
# income 평균 구하기
job_income = welfare.dropna(subset = ['job','income']).groupby('job', as_index = False).agg(mean_income = ('income','mean'))
job_income.head()
'''
                job  mean_income
0       가사 및 육아 도우미    92.455882
1               간호사   265.219178
2  감정∙기술영업및중개관련종사자    391.000000
3      건물 관리원 및 검표원   168.375000
4    건설 및 광업 단순 종사자   261.975000
'''

# 월급이 많은 직업 상위 10개 시각화

top10 = job_income.sort_values('mean_income',ascending = False).head(10)
'''
                        job  mean_income
98                의료 진료 전문가   781.000000
60                   법률 전문가   776.333333
140          행정 및 경영 지원 관리자   771.833333
63              보험 및 금융 관리자   734.750000
110        재활용 처리 및 소각로 조작원   688.000000
131     컴퓨터 하드웨어 및 통신공학 전문가   679.444444
24        기계∙로봇공학 기술자 및 시험원   669.166667
6         건설∙전기 및 생산 관련 관리자   603.083333
120               제관원 및 판금원   597.000000
100  의회 의원∙고위 공무원 및 공공단체 임원   580.500000
'''

import matplotlib as mpl

# 한글 폰트 설정
mpl.rc('font', family='AppleGothic')  # 맥 기본 한글 폰트
mpl.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

sns.barplot(data = top10, y = 'job', x = 'mean_income')


# 월급이 적은 직업 10개 시각화
bottom10 = job_income.sort_values('mean_income',ascending = True).head(10)
'''
                          job  mean_income
33   기타 돌봄∙보건 및 개인 생활 서비스 종사자    73.964286
34           기타 서비스 관련 단순 종사자    77.789474
128               청소원 및 환경미화원    88.461756
0                 가사 및 육아 도우미    92.455882
43            돌봄 및 보건 서비스 종사자   117.162338
97               음식 관련 단순 종사자   118.187500
39            농림∙어업 관련 단순 종사자   122.625000
139          학예사∙사서 및 기록물 관리사   140.000000
126         채굴 및 토목 관련 기능 종사자   140.000000
135              판매 관련 단순 종사자   140.909091
'''

sns.barplot(data = bottom10, y = 'job', x = 'mean_income')

sns.barplot(data = bottom10, y = 'job', x = 'mean_income').set(xlim=[0, 800])

# 성별에 따른 직업 빈도
# 남성 직업 빈도 상위 10개
# male 추출 query('컬럼명 = "male" ')
# job 결측치 제거
# job 빈도 구하기
# 내림차순 정렬
# 상위 10개

job_male = welfare.dropna(subset = 'job').query('sex == "male"').groupby('job', as_index = False).agg(n = ('job', 'count')).sort_values('n', ascending= False).head(10)
'''
                job    n
107       작물 재배 종사자  486
104         자동차 운전원  230
11        경영 관련 사무원  216
46        매장 판매 종사자  142
89           영업 종사자  113
127     청소원 및 환경미화원  109
4    건설 및 광업 단순 종사자   96
120    제조 관련 단순 종사자   80
3      건물 관리원 및 검표원   79
141          행정 사무원   74
'''
# 여성 직업 빈도 상위 10개
# female 추출 query('컬럼명 = "male" ')
# job 결측치 제거
# job 빈도 구하기
# 내림차순 정렬
# 상위 10개

job_female = welfare.dropna(subset = 'job').query('sex == "female"').groupby('job', as_index = False).agg(n = ('job', 'count')).sort_values('n', ascending= False).head(10)
'''
                  job    n
83          작물 재배 종사자  476
91        청소원 및 환경미화원  282
33          매장 판매 종사자  212
106       회계 및 경리 사무원  163
31    돌봄 및 보건 서비스 종사자  155
87       제조 관련 단순 종사자  148
73       음식 관련 단순 종사자  126
58        식음료 서비스 종사자  117
88                조리사  114
24   기타 서비스 관련 단순 종사자   97
'''


sns.barplot(data = job_male, y = 'job', x= 'n').set(xlim = [0, 500])

sns.barplot(data = job_female, y = 'job', x= 'n').set(xlim = [0, 500])

'''
종교 유무에 따른 이혼율
'''
# 종교 변수
# 혼인 여부 전처리

welfare['religion'].dtypes   # dtype('float64')

# 이상치 확인 
welfare['religion'].value_counts()
'''
religion
2.0    7815 # 없음
1.0    6603 # 있음
Name: count, dtype: int64
'''

welfare['religion'].isna().sum()

welfare['religion'] = np.where(welfare['religion'] == 1, 'reli_yes', 'reli_no')

welfare['religion'].value_counts()
'''
religion
reli_no     7815
reli_yes    6603
Name: count, dtype: int64
'''

# welfare에 religion 결합
welfare = welfare.merge(list_job, how = 'left', on = 'code_job')


# 혼인 여부 전처리
welfare['marriage_type'].dtypes

# 이상치 확인
welfare['marriage_type'].value_counts()
'''
marriage_type
1.0    7190 # 유배우
5.0    2357 # 미혼(18세 이상, 미혼모 포함)
0.0    2121 # 비해당 18세 미만
2.0    1954 # 사별
3.0     689 # 이혼
4.0      78 # 별거
6.0      29 # 기타(사망)
Name: count, dtype: int64
'''

# 이혼 여부 변수
welfare['marriage'] = np.where(welfare['marriage_type'] == 1, 'marragie',
                      np.where(welfare['marriage_type'] == 3, 'divorced',
                               'etc'))

'''
welfare['marriage']
Out[52]: 
0             etc
1             etc
2        divorced
3        marragie
4        marragie
  
14413    marragie
14414         etc
14415         etc
14416         etc
14417         etc
Name: marriage, Length: 14418, dtype: object
'''
welfare['marriage_type'].isna().sum()
# 이혼 여부 칼럼 생성
welfare['divorced'] = np.where(welfare['marriage_type'] == 3.0, 1, 0)
welfare['divorced'].value_counts()
'''
divorced
0    13729 # 이혼 x 
1      689 # 이혼
Name: count, dtype: int64
'''

# 이혼 여부별 빈도
# marriage 분리
# marriage 빈도 구하기
n_divorce = welfare.groupby('marriage', as_index = False).agg(n = ('marriage', 'count'))
'''
   marriage     n
0  divorced   689
1       etc  6539
2  marragie  7190
'''
sns.barplot(data = n_divorce, x = 'marriage', y = 'n')

# 종교 유무에 따른 이혼율 분석
# 종교 유무에 따른 이혼율표
# etc 제외
# religion 별 분리
# marriage 분리
# 비율 구하기

rel_div = welfare.query('marriage != "etc"').groupby('religion', as_index = False)['marriage'].value_counts(normalize = True)
'''
   religion  marriage  proportion
0   reli_no  marragie    0.905045
1   reli_no  divorced    0.094955
2  reli_yes  marragie    0.920469
3  reli_yes  divorced    0.079531
'''


divorce_rate = welfare.groupby('religion')['divorced'].mean() * 100
'''
religion
reli_no     4.913628
reli_yes    4.619113
Name: divorced, dtype: float64
'''
# 시각화
sns.barplot(x = divorce_rate.index, y = divorce_rate.values)
sns.countplot(data=welfare, x='religion', hue='divorced')


'''
지역별 연령대 비율
'''

# 지역 코드 데이터 전처리
# 연령대별 전처리는 전에 한 전처리 사용 age

# 지역 코드 전처리
welfare['code_region'].dtypes   # dtype('float64')
welfare['code_region'].value_counts()
'''
code_region
2.0    3246 # 수도권(인천/경기)
7.0    2466 # 광주/전남/전북/제주도
3.0    2448 # 부산/경남/울산
1.0    2002 # 서울
4.0    1728 # 대구/경북
5.0    1391 # 대전/충남
6.0    1137 # 강원/충북
Name: count, dtype: int64
'''
# 코드 한글로 변환
region_map = {
    1.0: '서울',
    2.0: '수도권(인천/경기)',
    3.0: '부산/경남/울산',
    4.0: '대구/경북',
    5.0: '대전/충남',
    6.0: '강원/충북',
    7.0: '광주/전남/전북/제주도'
}

# 지역명 변수 추가
list_region = pd.DataFrame({'code_region' : [1,2,3,4,5,6,7],
                            'region' : ['서울', '수도권', '부산/경남/울산',
                                        '대구/경북','대전/충남','강원/충북',
                                        '광주/전남/전북/제주도']})
'''
   code_region        region
0            1            서울
1            2           수도권
2            3      부산/경남/울산
3            4         대구/경북
4            5         대전/충남
5            6         강원/충북
6            7  광주/전남/전북/제주도
'''

# 지역명 변수 추가
welfare = welfare.merge(list_region, how = 'left', on = 'code_region')


welfare['code_region'] = welfare['code_region'].map(region_map)

welfare['code_region'].isna().sum() # 0

# 연령대별 분석
# 연령대 변수 만들기
welfare = welfare.assign(
    age_state = np.where(welfare['age'] < 10, '유아',
                np.where(welfare['age'] < 20, '10대',
                np.where(welfare['age'] < 30, '20대',
                np.where(welfare['age'] < 40, '30대',
                np.where(welfare['age'] < 50, '40대',
                np.where(welfare['age'] < 60, '50대',
                np.where(welfare['age'] < 70, '60대',
                np.where(welfare['age'] < 80, '70대',
                np.where(welfare['age'] < 90, '80대',
                np.where(welfare['age'] < 100, '90대','100세 이상')))))))))))



# 지역과 나이대 분석
welfare[['code_region', 'age_state']].head(10)
'''
  code_region age_state
0          서울       70대
1          서울       70대
2          서울       70대
3          서울       50대
4          서울       50대
5          서울       10대
6          서울       90대
7          서울       80대
8  수도권(인천/경기)       80대
9          서울       50대
'''

# 7개 지역별 연령대 분석, 평균 연령대도 출력하면 좋을듯

welfare.groupby('code_region')['age_state'].value_counts()
'''
code_region  age_state
강원/충북        70대          192
             80대          168
             50대          149
             60대          144
             40대          133

수도권(인천/경기)   60대          348
             30대          306
             80대          293
             유아           215
             90대           32
Name: count, Length: 73, dtype: int64
'''

# 지역별 연령대 계산
region_age_ratio = welfare.groupby('code_region')['age_state'].value_counts(normalize=True) * 100
'''
region      age_state
강원/충북       70대          16.886544
            80대          14.775726
            50대          13.104661
            60대          12.664908
            40대          11.697449
   
수도권(인천/경기)  60대          10.720887
            30대           9.426987
            80대           9.026494
            유아            6.623537
            90대           0.985829
Name: proportion, Length: 73, dtype: float64
'''
# 시각화
sns.countplot(data = welfare, x = 'code_region', hue = 'age_state')

'''
피벗 : 행과 열을 회전하여 표의 구성을 변경하는 작업.
        누적 그래프 형태로 시각화할때 사용
        1. 지역, 연령대, 비율 추출
            region_ageg [['region', 'ageg', proportion]]
        2. DataFrame.pivot()
        2.1 지역을 기준으로 : indec = 지역
        2.2 연령대별로 컬럼을 구성 : columns = 연령대
'''
# 1 
# pivot_df = region_ageg [['region', 'ageg', 'proportion']].pivot(index = 'region', columns = 'ageg', values = 'proportion')
# pivot_df.plot.barh(stacked = True) # 쌓을때

# reorder_df = pivot_df.sort_values('old')[['young','middle','old']]
# reorder_df.plot.barh(stacked = True) # 쌓을때




















# 📊 Korean Welfare Panel Data Analysis

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

## 🔍 소개

**Korean Welfare Panel Data Analysis**는 한국복지패널 데이터를 활용한 시각화 및 통계 분석 프로젝트입니다.  
소득, 연령, 직업, 만족도 등 다양한 변수 간 관계를 시각적으로 탐색하고 통계적으로 검증합니다.

이 프로젝트는 Python을 사용한 데이터 분석 예제로, 복지 관련 통계 분석 및 시각화에 관심 있는 분들에게 좋은 참고 자료가 됩니다.

---

## 🧩 주요 기능

- 📈 다양한 시각화: 막대그래프, 선형그래프, 박스플롯, 산점도 등
- 🧪 통계 분석: 그룹 간 평균 차이 검정을 위한 T-Test 구현
- 🧭 대화형 분석: Plotly 등을 이용한 interactive 시각화 제공

---

## 🛠️ 사용 기술

- Python 3.10+
- Pandas, Matplotlib, Seaborn
- Plotly, Scipy
- HTML export for visualization results

---

## 🗂️ 프로젝트 구조

```
📁 Korean-Welfare-Panel-master/
│
├── 📄 bar_plot.html         # 소득/복지 데이터의 막대그래프 시각화 결과
├── 📄 box_plot.html         # 변수 간 분포 비교를 위한 박스플롯 결과
├── 📄 interactive.py        # 대화형 시각화 구현 코드 (예: Plotly 등 활용)
├── 📄 korean_life2.py       # 주요 분석 수행 스크립트
├── 📄 line_plot.html        # 시계열 또는 추세선 기반 선형그래프 결과
├── 📄 scatter_plot.html     # 변수 간 상관관계 분석을 위한 산점도 결과
├── 📄 self.py               # 보조 분석 코드 또는 개별 실험용 스크립트
├── 📄 size_plot.html        # 데이터 크기를 시각적으로 표현한 결과 (버블 차트 등)
└── 📄 t_test.py             # 그룹 간 평균 차이 검정을 위한 T-test 코드
```

---

## 🚀 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/yourusername/Korean-Welfare-Panel.git
cd Korean-Welfare-Panel
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv
source venv/bin/activate  # 윈도우: venv\Scripts\activate
pip install -r requirements.txt  # (필요시 requirements.txt 생성)
```

### 3. 분석 스크립트 실행

```bash
python korean_life2.py
```

또는 각 개별 스크립트를 실행하여 결과를 확인할 수 있습니다.

---

## 📸 결과 예시

> 다음은 생성된 시각화 HTML 결과 중 일부입니다:

- ✅ [bar_plot.html](./bar_plot.html)
- ✅ [scatter_plot.html](./scatter_plot.html)
- ✅ [line_plot.html](./line_plot.html)

---

## 🧑‍💻 기여 방법

1. 이 레포지토리를 포크하세요.
2. 새로운 브랜치를 생성하세요: `git checkout -b feature/기능명`
3. 변경사항을 커밋하세요: `git commit -m "Add 기능"`
4. 브랜치에 푸시하세요: `git push origin feature/기능명`
5. Pull Request를 생성하세요.



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:13:09 2025

@author: oh
인터랙티브 시각화 : html 파일로 저장 => 웹브라우저로 실행
"""
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import webbrowser

mpg = pd.read_csv("./data/mpg.csv")

# 산점도 : px.scatter(data_frame = , x = '', y = '', color = '')

fig = px.scatter(data_frame= mpg, 
                 x = 'cty', 
                 y = 'hwy', 
                 color = 'drv')

# html로 저장
fig.write_html('scatter_plot.html')

webbrowser.open_new('scatter_plot.html')

# 막대 그래프
# 자동차 종류별 빈도
df = mpg.groupby('category', as_index = False).agg(n = ('category', 'count'))

fig = px.bar(data_frame= df, 
             x = 'category',
             y = 'n',
             color = 'category')

fig.write_html('bar_plot.html')
webbrowser.open_new('bar_plot.html')

# 선그래프 : px.line()

economics = pd.read_csv("./data/economics.csv")

fig = px.bar(data_frame= economics, 
             x = 'date',
             y = 'psavert')

fig.write_html('line_plot.html')

# 이상치를 확인할 수 있는 boxplot px.box()

fig = px.box(data_frame= mpg, 
             x = 'drv', 
             y = 'hwy', 
             color = 'drv')
fig.write_html('box_plot.html')


fig = px.scatter(data_frame= mpg, 
                 x = 'cty', 
                 y = 'hwy', 
                 color = 'drv',
                 width = 600, height= 400)
fig.write_html('size_plot.html')

































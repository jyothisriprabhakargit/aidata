from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import random
#import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
#import st_static_export as sse

import warnings
warnings.filterwarnings("ignore")

#st.data = pd.read_csv('sensordata.csv')

@st.cache_data
def load_data(path: str):
	data = pd.read_excel(path)
	return data


data = load_data("/workspaces/ai-analytics/hhh.xlsx")

#st.write("shape of hole dataset")
#data.shape

#data.head()


st.sampled_df = data[(data['id1'] % 10) == 0]
st.write("shape of sample data set")
st.sampled_df.shape


st.sampled_df.describe().transpose()

#st.sampled_df[st.sampled_df['temprature1'] == 0].shape

#st.sampled_df[st.sampled_df['air_quality1'] == 0].shape

#st.sampled_df[st.sampled_df['temprature2'] == 0].shape

#st.sampled_df[st.sampled_df['air_quality2'] == 0].shape


rows_before = st.sampled_df.shape[0]
sampled_df = st.sampled_df.dropna()
rows_after = sampled_df.shape[0]

#rows_before - rows_after
st.write("head of data set")
sampled_df.columns

features1 = ['temprature1', 'temprature2']
features2 = ['air_quality1', 'air_quality2']
#features = ['temprature1', 'air_quality1','temprature2', 'air_quality2']

select_df1 = sampled_df[features1]
select_df2 = sampled_df[features2]
st.write("head of Temprature")
select_df1.columns
st.write("head of Air_quality")
select_df2.columns
st.write("Data of Temprature")
select_df1
st.write("Data of Air_quality")
select_df2
X = StandardScaler().fit_transform(select_df1)
st.write("StandardScaler Data of Temprature")
X
Y = StandardScaler().fit_transform(select_df2)
st.write("StandardScaler Data of Air_quality")
Y
kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)

st.write("StandardScaler Data after kmeans of Temprature")
centers1 = model.cluster_centers_
centers1

def pd_centers(featuresUsed, centers1):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers1)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P



def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(6,6)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')

      

P = pd_centers(features1, centers1)
st.write("prediction of Temprature")
P

st.write("StandardScaler Data after kmeans of Air_quality")
centers2 = model.cluster_centers_
centers2

def pd_centers1(featuresUsed, centers2):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers1)]

	# Convert to pandas data frame for plotting
	P1 = pd.DataFrame(Z, columns=colNames)
	P1['prediction'] = P1['prediction'].astype(int)
	return P1



def parallel_plot1(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(6,6)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')

P1 = pd_centers1(features2, centers2)
st.write("prediction of Air_quality")
P1

st.write("Parallel_plot of Temprature")
st.pyplot(parallel_plot(P[P['temprature2'] < 0.5]))

st.write("Parallel_plot of Air_quality")
st.pyplot(parallel_plot(P1[P1['air_quality1'] > 0.5]))





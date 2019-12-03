import gdelt
import pandas as pd
import numpy as np
import platform
import multiprocessing
from datetime import datetime
import calendar
import os
import networkx as nx
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from scipy.stats import norm

def create_database(in_folder='C:/Users/ssq10/Documents/Classes/ISE 599/Project/New Data/', GoldStein=3, AvgTone=5): # replace this path with your own after you've unzipped the input file.
    directory = in_folder
    master_col = col_namer()
    master_list = []
    for f in os.listdir(directory):
        if "East_Afr" in f:
            data = pd.read_csv(directory + f)
            df = edgifier(data)
            eri_df, eth_df, sdn_df, ssd_df = country_split(df)
            countries = [eri_df, eth_df, sdn_df, ssd_df]
            cntry_names = ['COD','ETH','CAF','SSD']

            for a,b in zip(countries,cntry_names):
                G_list = network_transform(a, GS=GoldStein, AT=AvgTone)
                mstr = analysis_row_creator(yrmo=f,country=b,Graphs=G_list, dataframe=data)
                master_list.append(mstr)
    df = pd.DataFrame(master_list,columns=master_col)
    return(df)


# Only Nodes with Edges (Clean)
def edgifier(data):
    data['Actor1Code'].replace('', np.nan, inplace=True)
    data['Actor2Code'].replace('', np.nan, inplace=True)
    data.dropna(subset=['Actor1Code'], inplace=True)
    data.dropna(subset=['Actor2Code'], inplace=True)
    return(data)


# Split into countries - Plan: Split, Transform,  Merge
def country_split(data):
    eri_df = data[(data.Actor1CountryCode == 'CAF') | (data.Actor2CountryCode == 'CAF')]
    eth_df = data[(data.Actor1CountryCode == 'ETH') | (data.Actor2CountryCode == 'ETH')]
    sdn_df = data[(data.Actor1CountryCode == 'COD') | (data.Actor2CountryCode == 'COD')]
    ssd_df = data[(data.Actor1CountryCode == 'SSD') | (data.Actor2CountryCode == 'SSD')]
    return(eri_df, eth_df, sdn_df, ssd_df)

def network_transform(data, GS = 3, AT = 5):
    gp = data[(data.GoldsteinScale > GS) & (data.AvgTone > AT)]
    gi = data[(data.GoldsteinScale > GS) & (data.AvgTone < AT) & (data.AvgTone > -AT)]
    gn = data[(data.GoldsteinScale > GS) & (data.AvgTone < -AT)]
    np = data[(data.GoldsteinScale > -GS) & (data.GoldsteinScale < GS) & (data.AvgTone > AT)]
    ni = data[
        (data.GoldsteinScale > -GS) & (data.GoldsteinScale < AT) & (data.AvgTone < AT) & (data.AvgTone > -AT)]
    nn = data[(data.GoldsteinScale > -GS) & (data.GoldsteinScale < AT) & (data.AvgTone < -AT)]
    bp = data[(data.GoldsteinScale < -GS) & (data.AvgTone > AT)]
    bi = data[(data.GoldsteinScale < -GS) & (data.AvgTone < AT) & (data.AvgTone > -AT)]
    bn = data[(data.GoldsteinScale < -GS) & (data.AvgTone < -AT)]

    all_df = [gp, gi, gn, np, ni, nn, bp, bi, bn]

    # Filling some lists!
    gp_n, gi_n, gn_n = [], [], []
    np_n, ni_n, nn_n = [], [], []
    bp_n, bi_n, bn_n = [], [], []
    node_list = [gp_n, gi_n, gn_n, np_n, ni_n, nn_n, bp_n, bi_n, bn_n]

    for i, j in zip(all_df, node_list):
        j = i.dropna(subset=['Actor1Code']).Actor1Code
        j.append(i.dropna(subset=['Actor2Code']).Actor2Code)
        j = j.unique()

    G_gp, G_gi, G_gn = nx.MultiDiGraph(), nx.MultiDiGraph(), nx.MultiDiGraph()
    G_np, G_ni, G_nn = nx.MultiDiGraph(), nx.MultiDiGraph(), nx.MultiDiGraph()
    G_bp, G_bi, G_bn = nx.MultiDiGraph(), nx.MultiDiGraph(), nx.MultiDiGraph()

    G_list = [G_gp, G_gi, G_gn, G_np, G_ni, G_nn, G_bp, G_bi, G_bn]

    for i, j, k in zip(node_list, G_list, all_df):
        j.add_nodes_from(i)

        j.add_edges_from(list(zip(k.Actor1Code, k.Actor2Code)),
                         relation=list(k.QuadClass))
    return(G_list)

def analysis_row_creator(yrmo, country, Graphs, dataframe):
    n_cnt = []
    #e_cnt = []
    n_deg = []
    moyr = yrmo[9:-7]

    for i in Graphs:
        nodes = i.number_of_nodes()
        edges = i.number_of_edges()
        n_cnt.append(nodes)
        #e_cnt.append(edges)
        if nodes > 0:
            n_deg.append(edges / nodes)  # directed
        else: n_deg.append(0)

    mstr_list = [country, moyr]
    for m in n_cnt:
        mstr_list.append(m)
    #for n in e_cnt:
        #mstr_list.append(n)
    for o in n_deg:
        mstr_list.append(o)

    df = dataframe[(dataframe.Actor1CountryCode == country)]
    count = pd.DataFrame(df['EventRootCode'].value_counts()).transpose()
    tester = pd.DataFrame(columns=list(range(1, 21))).append(count).fillna(0)
    root_cnt = list(tester.values[0])

    for p in root_cnt:
        mstr_list.append(p)

    return(mstr_list)



def col_namer():
    list_col_n = ['G-P Nodes', 'G-I Nodes', 'G-N Nodes', 'N-P Nodes', 'N-I Nodes',
                'N-N Nodes', 'B-P Nodes', 'B-I Nodes', 'B-N Nodes']
    #list_col_e = ['G-P Edges', 'G-I Edges', 'G-N Edges', 'N-P Edges', 'N-I Edges',
                #'N-N Edges', 'B-P Edges', 'B-I Edges', 'B-N Edges']
    list_col_d = ['G-P Degree', 'G-I Degree', 'G-N Degree', 'N-P Degree', 'N-I Degree',
                'N-N Degree', 'B-P Degree', 'B-I Degree', 'B-N Degree']
    mstr_col_list = ['Country_Code','Year/Month']
    loop_col_list = [list_col_n, list_col_d] #  , list_col_e <--- insert before list_col_d if needed
    for m in list_col_n:
        mstr_col_list.append(m)
    #for n in list_col_e:
        #mstr_col_list.append(n)
    for o in list_col_d:
        mstr_col_list.append(o)
    for p in list(range(1,21)):
        mstr_col_list.append(p)
    return(mstr_col_list)


def ACLED_data(in_folder='C:/Users/ssq10/Documents/Classes/ISE 599/Project/New Data/2015-03-01-2019-11-30-Central_African_Republic-Democratic_Republic_of_Congo-Ethiopia-South_Sudan.csv'):
    data = pd.read_csv(in_folder)
    data['event_date'] = pd.to_datetime(data['event_date'])
    data = data[data['event_type'] == 'Battles'] # Filters for just Battle Fatalities - Remove for fatalities in general
    data['year'], data['month'] = data['event_date'].dt.year, data['event_date'].dt.month
    fat_mo = pd.DataFrame(data.groupby(['year', 'month', 'country'])['fatalities'].agg('sum')).unstack(fill_value=0).stack().reset_index()
    conditions = [(fat_mo['country'] == 'Democratic Republic of Congo'), (fat_mo['country'] == 'Central African Republic'),
                  (fat_mo['country'] == 'South Sudan'), (fat_mo['country'] == 'Ethiopia')]
    choices = ['COD', 'CAF', 'SSD', 'ETH']
    fat_mo['Country_Code'] = np.select(conditions, choices)
    fat_mo['Year/Month'] = fat_mo['year'].map(str) + ' ' + fat_mo['month'].map(str)
    fat_mo['Year/Month'] = pd.to_datetime(fat_mo['Year/Month'], format = '%Y %m') - pd.DateOffset(months=1)
    fat_mo['Year/Month'] = fat_mo['Year/Month'].map(str)
    fat_mo['Year/Month'] = fat_mo['Year/Month'].str[0:4] + ' ' + fat_mo['Year/Month'].str[5:7]
    fat_mo = fat_mo.drop(columns=['country', 'year'])

    return(fat_mo)



def model_compiler():
    df_list = ['8-3','7-3','6.5-3','6-3.5', '6-3']
    GS_list = [8, 7, 6.5, 6, 6]
    AT_list = [3, 3, 3, 3.5, 3]
    Rand_Forest = []
    RF_Bagged = []
    Dec_Tree = []
    Ridge_Reg = []
    Lasso_Reg = []

    for i,j in zip(GS_list,AT_list):
        gdelt_df = create_database(GoldStein=i, AvgTone=j)
        acled_df = ACLED_data()
        df = pd.merge(gdelt_df, acled_df, on = ['Year/Month','Country_Code'])

        df = df[(df.Country_Code == 'COD') | (df.Country_Code == 'SSD') | (df.Country_Code == 'CAF') | (df.Country_Code == 'ETH')]
        df2 = df.drop(columns=['B-P Nodes', 'B-P Degree', 'Year/Month'])
        df3 = df2.dropna()
        df3 = pd.get_dummies(df3, columns=['Country_Code', 'month'])
        X = df3.drop(columns='fatalities')
        y = df3.fatalities
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            random_state=2)

        kfold = KFold(n_splits=10)
        m1 = RandomForestRegressor(n_estimators = 80,max_features = 10, max_depth = 6,random_state=0)
        m2 = RandomForestRegressor(max_features=5,n_estimators = 500)
        m3 = DecisionTreeRegressor(max_depth=5,max_features=4,random_state=0)

        alphas = np.linspace(10,-2,100)
        alphas = 10**alphas
        m4_pre = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error',
                 normalize = True)
        m4_pre.fit(X_train,y_train)
        m4 = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error',
                 normalize = True)

        m5 = LassoCV(alphas=alphas,cv=10,max_iter=10000,normalize=True)

        m1_res = cross_val_score(m1,X,y,cv=kfold,scoring='neg_mean_squared_error')
        m2_res = cross_val_score(m2, X, y, cv=kfold, scoring='neg_mean_squared_error')
        m3_res = cross_val_score(m3, X, y, cv=kfold, scoring='neg_mean_squared_error')
        m4_res = cross_val_score(m4, X, y, cv=kfold, scoring='neg_mean_squared_error')
        m5_res = cross_val_score(m5, X, y, cv=kfold, scoring='neg_mean_squared_error')

        Rand_Forest.append(-m1_res.mean())
        RF_Bagged.append(-m2_res.mean())
        Dec_Tree.append(-m3_res.mean())
        Ridge_Reg.append(-m4_res.mean())
        Lasso_Reg.append(-m5_res.mean())

    models_df = pd.DataFrame()
    models_df['GS_and_AT'] = df_list
    models_df['Random_Forest'] = Rand_Forest
    models_df['RF_Bagged'] = RF_Bagged
    models_df['Dec_Tree'] = Dec_Tree
    models_df['Ridge_Regress'] = Ridge_Reg
    models_df['Lasso_Regress'] = Lasso_Reg

    return(models_df)


model_df3 = model_compiler() # Number to beat - 10275.49
mean_squared_error(y,[110.68]*160) # Number to Beat!


def combine_dataframes(GS=6, AT=3):
    gdelt_df = create_database(GoldStein=GS, AvgTone=AT)
    acled_df = ACLED_data()
    df = pd.merge(gdelt_df, acled_df, on=['Year/Month', 'Country_Code'])
    return(df)



df.to_csv('C:/Users/ssq10/Documents/Classes/ISE 599/Project/Analysis/battle_data.csv')


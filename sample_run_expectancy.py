import streamlit as st

# numpyとpandasを使います. 計算はほぼpandas
import csv
import numpy as np
import pandas as pd

st.write('# 得点期待値をPythonで算出するサンプル')

st.write('## データ読み込みと下処理')

# Dataframeで使うカラムとデータ型の定義
DF_COLUMNS = {
    'HOME_SCORE_CT': np.int32,
    'AWAY_SCORE_CT': np.int32,
    'GAME_ID': str,
    'INN_CT': np.int32,
    'BAT_HOME_ID': str,
    'BAT_DEST_ID': np.int32,
    'RUN1_DEST_ID': np.int32,
    'RUN2_DEST_ID': np.int32,
    'RUN3_DEST_ID': np.int32,
    'BASE1_RUN_ID': str,
    'BASE2_RUN_ID': str,
    'BASE3_RUN_ID': str,
    'OUTS_CT': np.int32,
    'EVENT_OUTS_CT': np.int32,
}

# CSVからDataframeを作る（headerとbody別れてるのでややこしいことになってます）
with open('./data/fields.csv', 'r') as f_header:
    headers = [row.get('Header') for row in csv.DictReader(f_header)]
with open('./data/all2016.csv', 'r') as f_body:
    values = [dict(zip(headers, row)) for row in csv.reader(f_body)]
df = pd.DataFrame(values, columns=DF_COLUMNS.keys()).astype(DF_COLUMNS)
st.write(df.head())


st.write('## いよいよ計算')

# 得点を表すカラムを追加（ホームとアウェイで得点が別れてるので一緒にする）
df['RUNS'] = df['HOME_SCORE_CT'] + df['AWAY_SCORE_CT']

# イニングのKey値を表すカラム
df['HALF_INNING'] = df['GAME_ID'].astype(str) + df['INN_CT'].astype(str) + df['BAT_HOME_ID'].astype(str)

# 得点イベント時の総得点
def _run_scored(dests: str) -> int:
    """
    Calc Run Scored
    :param dests: Dests Strings
    :return: Run Scored
    """
    runs_scored = 0
    for dest in dests:
        if int(dest) > 3:
            runs_scored += 1
    return runs_scored
df['DESTS'] = df['BAT_DEST_ID'].astype(str) + df['RUN1_DEST_ID'].astype(str) + \
              df['RUN2_DEST_ID'].astype(str) + df['RUN3_DEST_ID'].astype(str)
df['RUNS_SCORED'] = df['DESTS'].map(_run_scored)

# ランナーの人数×アウトカウントでの状況を中間カラムとして設定
def _on_base(base_run_id) -> str:
    """
    Exists Runner
    :param base_run_id: retrosheet base_run_oid
    :return: '1' or '0'(1:True, 0:False)
    """
    if type(base_run_id) == float and math.isnan(base_run_id):
        return '0'
    elif type(base_run_id) == str and len(base_run_id) > 0:
        return '1'
    return '0'

df['BASES'] = df['BASE1_RUN_ID'].map(_on_base) + df['BASE2_RUN_ID'].map(_on_base) \
              + df['BASE3_RUN_ID'].map(_on_base)
df['STATE'] = df['BASES'].astype(str) + ' ' + df['OUTS_CT'].astype(str)

# プレー後の状況を表すカラム

def _new_base(dests: str) -> str:
    """
    Create New Base State
    :param dests: Dests
    :return: New Base
    """
    bat, run1, run2, run3 = int(dests[:1]), int(dests[1:2]), int(dests[2:3]), int(dests[3:4])
    nrunner1, nrunner2, nrunner3 = '0', '0', '0'
    if run1 == 1 or bat == 1:
        nrunner1 = '1'
    if run1 == 2 or run2 == 2 or bat == 2:
        nrunner2 = '1'
    if run1 == 3 or run2 == 3 or run3 == 3 or bat == 3:
        nrunner3 = '1'
    return f"{nrunner1}{nrunner2}{nrunner3}"

def _new_state(dest_outs: str) -> str:
    """
    Create New State
    :param dest_outs: Dests & Outs
    :return: New State
    """
    list_dest_outs = dest_outs.split('_')
    return f"{_new_base(list_dest_outs[0])} {list_dest_outs[1]}"


df['OUTS_CNT'] = df['OUTS_CT'] + df['EVENT_OUTS_CT']
df['DESTS_OUTS'] = df['DESTS'] + '_' + df['OUTS_CNT'].astype(str)
df['NEW_BASE'] = df['DESTS'].map(_new_base)
df['NEW_STATE'] = df['DESTS_OUTS'].map(_new_state)

# ここまでの途中経過
st.write('### 途中経過')
st.write(df.head())

# 用が済んだカラムを捨てます
df.drop(columns=['DESTS', 'OUTS_CNT', 'DESTS_OUTS'], inplace=True)
st.write('### カラムを捨てた後')
st.write(df.head())

st.write('## 算出する')

# 中間結果としてイニングごとのruns_scoredを出す
df_runs_scored_inning = df[['RUNS_SCORED', 'HALF_INNING']].groupby('HALF_INNING', as_index=False).sum()

# さらに先頭の結果のみ抽出
df_runs_scored_start = df[['RUNS', 'HALF_INNING']].groupby('HALF_INNING', as_index=False).first()

# 上記の中間DataFrameをmerge
df_max = pd.merge(df_runs_scored_start, df_runs_scored_inning, how='inner', on='HALF_INNING')
df_max['MAX_RUNS'] = df_max['RUNS'] + df_max['RUNS_SCORED']
df_roi = pd.merge(df, df_max[['HALF_INNING', 'MAX_RUNS']], how='inner', on='HALF_INNING')

# RUNS ROI算出
df_roi['RUNS_ROI'] = df_roi['MAX_RUNS'] - df_roi['RUNS']
st.write('### RUNS ROI計算後')
st.write(df_roi.head())

# 必要なレコードのみに絞る. 具体的には「状況が変わった or 得点が入った」イベントのみ残す
df_roi.query("STATE != NEW_STATE or RUNS_SCORED > 0", inplace=True)

# イニングごとにサマってアウトを合計, カラム名も変更
df_data_outs = df_roi[['HALF_INNING', 'EVENT_OUTS_CT']].groupby('HALF_INNING', as_index=False).sum()
df_data_outs.rename(columns={'EVENT_OUTS_CT': 'OutsInning'}, inplace=True)

# アウトになったデータをMerge
df_run_ex = pd.merge(df_roi, df_data_outs, how='inner', on='HALF_INNING')

# 必要なものだけに絞ってデータ的には完成
df_run_ex = df_run_ex.query("OutsInning == 3")[['RUNS_ROI', 'STATE']].groupby('STATE', as_index=False).mean().round(2)
st.write('### 完成したデータ')
df_run_ex

st.write('## アウトプット用のデータにする')

# ランナー状況の記号
RUNNER_STATUS_COLUMNS = {
    '000': {'key': '___', 'status': 'no runner'},  # ランナー無し
    '001': {'key': '__3', 'status': '3B'},  # 3塁
    '010': {'key': '_2_', 'status': '2B'},  # 2塁
    '011': {'key': '_23', 'status': '2B3B'},  # 2塁3塁
    '100': {'key': '1__', 'status': '1B'},  # 1塁
    '101': {'key': '1_3', 'status': '1B3B'},  # 1塁3塁
    '110': {'key': '12_', 'status': '1B2B'},  # 1塁2塁
    '111': {'key': '123', 'status': '1B2B3B'},  # 満塁
}

# 列を分割
df_runs_colums = df_run_ex['STATE'].str.split(' ', expand=True)
for key, value in RUNNER_STATUS_COLUMNS.items():
    df_runs_colums[0].replace(key, value.get('key'), inplace=True)
    
# アウトおよび走者を再び代入
df_run_ex['outs'] = df_runs_colums[1]
df_run_ex['runner'] = df_runs_colums[0]

# STATEはいらなくなったので消す
del df_run_ex['STATE']

# 表示用のDataFrameを作って完成
df_view = pd.pivot_table(df_run_ex, index='runner', columns='outs')
st.write('### 表示用のデータ')
df_view

dict_roi = df_view.to_dict()
dict_view = {}
dict_view['0outs'] = dict_roi[('RUNS_ROI', '0')]
dict_view['1outs'] = dict_roi[('RUNS_ROI', '1')]
dict_view['2outs'] = dict_roi[('RUNS_ROI', '2')]
df = pd.DataFrame(dict_view)
df

import plotly.graph_objects as go

st.write('## ヒートマップ')
fig = go.Figure(data=go.Heatmap(
                   z=df,
                   x=['ノーアウト', '1アウト', '2アウト'],
                   y=['満塁', '一塁二塁', '一塁三塁', '一塁', '二塁三塁', '二塁', '三塁', 'ランナー無し'],
                   hoverongaps = True))
fig
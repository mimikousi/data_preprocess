import pandas as pd 
import glob

# CSVファイルがあるディレクトリのパスを指定
path = r"C:\path\to\your\data" 

# 指定したディレクトリ内の全CSVファイルのパスを取得
all_files = glob.glob(path + "/*.csv") 

# ディレクトリ内の各CSVファイルに対してループ
df_ = pd.DataFrame()
for filename in all_files:
    df = pd.read_csv(filename, index_col=0, header=0, encoding='shift-jis') 
    df_ = pd.concat([df_, df], axis=0)

# 結合したデータをpickleファイルとして保存
df_.to_pickle('total_dataframe.pkl') 

# インデックスをdatetime型に変換
df_.index = pd.to_datetime(df_.index)

# インデックスを時系列順に並べ替え
df_ = df_.sort_index()

# 必要であれば、結果を別のファイルに保存
df_.to_pickle('sorted_dataframe.pkl')
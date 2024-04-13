import numpy as np

# 赤池情報量基準
def AIC(x, y):
    beta = np.linalg.solve(x.T @ x, x.T @ y) # 回帰係数を求めます。
    y_pred = x @ beta # 予測値を求めます。
    resid = y - y_pred # 残渣（実データ－予測値）を求めます。
    rss = resid.T @ resid # 残渣の二乗和を求めます。
    AIC = len(y) *np.log(rss/len(y)) + (x.shape[1])*2 # AICを求めます。
    return AIC

# ベイズ情報量基準
def BIC(x, y):
    beta = np.linalg.solve(x.T @ x, x.T @ y) # 回帰係数を求めます。
    y_pred = x @ beta # 予測値を求めます。
    resid = y - y_pred # 残渣（実データ－予測値）を求めます。
    rss = resid.T @ resid # 残渣の二乗和を求めます。
    BIC = len(y) *np.log(rss/len(y)) + (x.shape[1])*np.log(len(y)) # BICを求めます。
    return BIC

# 変数選択後のサンプルデータから回帰係数を求めます。
def forward_stepwise_result(x, y):
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    return beta

# forward_stepwise selection
def forward_stepwise(x, y, method='BIC'):
    """
    forward stepwise（変数増加法）で変数選択を行います。
        
    パラメータ
    ----------
        x :説明変数のデータセット 
        y :目的変数のデータセット
        method: 'AIC' or 'BIC'. Defaults to 'BIC'.
    戻り値
    ----------
        beta:回帰係数（最小二乗法）
        included_column:選択されたカラム名
        result_min: AICかBICの最終結果（最小値）
    """
    # xに要素１のintercept列を追加（0列目）
    x_tmp = np.concatenate(
        [np.reshape(np.ones(x.shape[0]), (x.shape[0],1)), x], axis=1
    ) # 要素１のarrayをn行１列にreshapeし、xと結合
    included = list([0]) #intercept列のみ（初期値）
    while True:
        changed = False
        excluded = list(set(np.arange(x_tmp.shape[1]))-set(included) ) #　除外した（未採用）列の番号list
        result = np.zeros(len(excluded)) #AIC,BICの計算結果を格納する変数
        if method == 'AIC':
            base_result = AIC(x_tmp[:, included], y) # 採用列のデータのみでAICを計算
        else:
            base_result = BIC(x_tmp[:, included], y) # 採用列のデータのみでBICを計算
        print('Baseline model with {}:{:}'.format(method, base_result))
        j = 0
        for new_column in excluded:
            if method == 'AIC':
                result[j] = AIC(x_tmp[:, included + [new_column]], y) # 採用列以外の列を追加してAICを順番に計算する
            else:     
                result[j] = BIC(x_tmp[:, included + [new_column]], y) # 採用列以外の列を追加してBICを順番に計算する
            # print('Add:{:15} with {}: {:}'.format(excluded[j], method,result[j])) # コメントアウト
            j += 1
        if result.min() < base_result:
            best_feature = excluded[result.argmin()] # argmin():最小値を取るインデックスを返す
            included.append(best_feature) # 最小値を取る列数をincluded（採用列）に追加
            changed = True
            print('Finally Add {:15} with {} {:}'.format(best_feature, method, result.min()))
        if not changed: #changed=Falseのとき
            print('Any variables does not added, stop forward stepwise regression')
            break
            # end while
                
    # final resultの回帰係数を返す
    beta = np.reshape(np.zeros(x_tmp.shape[1]), (x_tmp.shape[1],1)) # 回帰係数を格納する変数
    beta = forward_stepwise_result(x_tmp[:, included], y) # 回帰係数を計算
    included.pop(0) # intercept列は除去
    included_r = [ i - 1 for i in included] # xのcolumn numberに変更（included - 1）
    included_column = x.columns[[included_r]] # 採用した列名を取得
    result_min = base_result # 最小値を取得
    
    return beta, included_column, result_min
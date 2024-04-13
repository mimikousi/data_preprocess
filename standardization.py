import numpy as np

def autoscaling(X):
    """
    データ行列を標準化します
    Parameters
    -----------
    X : pandas.Dataframe or pandas.Series
        データ行列
    Returns
    -----------
    Xscale : pandas.Dataframe or pandas.Series
        標準化後のデータ行列
    meanX : pandas.Series or numpy.float64
        平均値ベクトル
    stdX : pandas.Series or numpy.float64
        標準偏差ベクトル
    """
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0, ddof=1) # ddof=1:不偏分散
    Xscale = (X - meanX) / stdX
    return Xscale, meanX, stdX

def scaling(x, meanX, stdX):
    """
    データ行列の平均と標準偏差からサンプルを標準化します
    
    Parameters
    -----------
    x : pandas.Dataframe or pandas.Series
        標準化したいサンプル
    meanX : pandas.Series or numpy.float64
        平均値ベクトル
    stdX : pandas.Series or numpy.float64
        標準偏差ベクトル
    Returns
    -----------
    xscale : pandas.Dataframe or pandas.Series
        標準化後のサンプル
    """
    xscale = (x - meanX) / stdX
    return xscale

def rescaling(xscale, meanX, stdX):
    """
    標準化されたサンプルを元のスケールに戻します
    Parameters
    -----------
    xscale : pandas.Dataframe or pandas.Series
        標準化後のサンプル
    meanX : pandas.Series or numpy.float64
        平均値ベクトル
    stdX : pandas.Series or numpy.float64
        標準偏差ベクトル
    Returns
    -----------
    x : pandas.Dataframe or pandas.Series
        元のスケールのサンプル
    """
    x = stdX * xscale + meanX
    return x
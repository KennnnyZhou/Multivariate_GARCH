import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import math
from arch import arch_model
from sklearn.utils.testing import ignore_warnings
from datetime import datetime

plt.style.use('seaborn')
import warnings

warnings.filterwarnings('ignore')


class OPT:
    def __init__(self, data):
        self.data = data
        self.tickers = data.columns[0:-2]

    def regression(self, x, y):
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        results = model.summary()
        return model, results

    def regression_res(self, ret_0, ret_1):
        tickers = self.tickers
        res_0 = []
        res_1 = []
        new_tickers = []
        for i in tickers:
            y1 = ret_0[i].dropna()
            y2 = ret_1[i].dropna()
            if len(y1) > 0 and len(y2) > 0:
                x1 = ret_0.loc[y1.index]['SPX Index']
                res_0.append(self.regression(x1, y1)[0])
                x2 = ret_1.loc[y2.index]['SPX Index']
                res_1.append(self.regression(x2, y2)[0])
                new_tickers += [i]
        data = ret_0[new_tickers]
        n = data.loc[:, :'GILD'].shape[1]
        return res_0, res_1, new_tickers, n

    def get_beta(self, regression_res_0, regression_res_1):
        beta_0 = []
        beta_1 = []
        for i in range(len(regression_res_0)):
            beta_0.append(regression_res_0[i].params[1])
            beta_1.append(regression_res_1[i].params[1])
        return beta_0, beta_1

    def pred_ret(self, regression_res_0, regression_res_1, ret=0.001):
        ret_list_0 = []
        for i in range(len(regression_res_0)):
            ret_list_0.append(regression_res_0[i].predict([0, ret])[0])
        ret_list_1 = []
        for i in range(len(regression_res_1)):
            ret_list_1.append(regression_res_1[i].predict([0, ret])[0])
        return ret_list_0, ret_list_1

    def pred_cov(self, stock_returns, period):
        stock_returns = stock_returns.dropna()
        warnings.filterwarnings('ignore')
        returns = np.array(list(stock_returns.values)).T
        initial_numberof_returns = returns[1].size
        esubt = []
        q_list = []
        # initial values
        cov_matrix = np.cov(returns)
        diag = cov_matrix.diagonal()
        diag_matrix = np.diag(np.diag(cov_matrix))
        garch11 = arch_model(diag, p=1, q=1)
        res = garch11.fit(update_freq=10, disp='off')
        alpha = res.params[2]
        beta = res.params[3]

        esubt.append(np.matmul(np.linalg.pinv(diag_matrix), returns))
        q_list.append(np.ones((len(stock_returns.columns), len(stock_returns.columns))))

        pred_cov = []
        new_returns = []

        # garch 1,1
        def simulate_GARCH(T, a0, a1, b1, sigma1):
            # Initialize our values
            X = np.ndarray(T)
            sigma = np.ndarray(T)
            sigma[0] = sigma1

            for t in range(1, T):
                # Draw the next x_t
                X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
                # Draw the next sigma_t
                sigma[t] = math.sqrt(a0 + b1 * sigma[t - 1] ** 2 + a1 * X[t - 1] ** 2)

            X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)

            return X, sigma

        # DCC Garch
        def dccGARCH(returns):
            cov_matrix = np.cov(returns)
            diag = cov_matrix.diagonal()
            diag_matrix = np.diag(np.diag(cov_matrix))

            garch11 = arch_model(diag, p=1, q=1)
            res = garch11.fit(update_freq=10, disp='off')
            alpha = res.params[2]
            beta = res.params[3]

            esubt.append(np.matmul(np.linalg.pinv(diag_matrix), returns))
            qdash = np.cov(esubt[-2])
            q_list.append(((1 - alpha - beta) * qdash) +
                          (alpha * np.matmul(esubt[-2], esubt[-2].T)) + ((beta) * q_list[-1]))

            qstar = np.diag(np.diag(cov_matrix))
            qstar = np.sqrt(qstar)

            r = np.matmul(np.linalg.pinv(qstar), q_list[-1], np.linalg.pinv(qstar))

            pred_cov_temp = np.matmul(diag_matrix, r)
            pred_cov.append(np.matmul(pred_cov_temp, diag_matrix))

        periods_to_predict = period

        for n in returns:
            garch11 = arch_model(n, p=1, q=1)
            res = garch11.fit(update_freq=10, disp='off')
            omega = res.params[1]
            alpha = res.params[2]
            beta = res.params[3]
            sig_ma = np.std(n)
            return_forecast, sigma_forecast = simulate_GARCH(periods_to_predict, omega, alpha, beta, sig_ma)
            n = np.append(n, return_forecast)
            new_returns.append(n)

        for i in range(initial_numberof_returns, new_returns[1].size):
            dccGARCH(returns[:, 0:i])

        cov = pred_cov[-1]

        return cov

    def optimization(self, mean_ret_0, mean_ret_1, cov, a, n):
        mean_ret_0 = (np.array(mean_ret_0) * 0.1).tolist()
        mean_ret_1 = (np.array(mean_ret_1) * 0.1).tolist()
        G = np.matrix(np.ones((2, len(mean_ret_0))))
        G[1, n + 1:] = 0
        U, d, V = np.linalg.svd(cov)
        U = np.matrix(U)
        V = np.matrix(V)
        D = np.matrix(np.diag(d))

        C_inv = U * D.I * V
        c = np.matrix([1, 0.65 * ((len(mean_ret_0) / 2 - n) + 1)]).T

        R0 = np.matrix(mean_ret_0).T
        R1 = np.matrix(mean_ret_1).T
        Lambda = (G * C_inv * G.T).I * (G * C_inv * (R0 - R1) - 2 * a * c)
        w = 1 / 2 / a * C_inv * ((R0 - R1) - G.T * Lambda)

        return w

    def optimization2(self, mean_ret_0, mean_ret_1, cov, a, n):
        mean_ret_0 = (np.array(mean_ret_0) * 0.1).tolist()
        mean_ret_1 = (np.array(mean_ret_1) * 0.1).tolist()
        G = np.matrix(np.ones((2, len(mean_ret_0))))
        G[1, :n] = 0
        U, d, V = np.linalg.svd(cov)
        U = np.matrix(U)
        V = np.matrix(V)
        D = np.matrix(np.diag(d))

        C_inv = U * D.I * V
        c = np.matrix([1, 0.65 * ((n - len(mean_ret_0) / 2) + 1)]).T

        R0 = np.matrix(mean_ret_0).T
        R1 = np.matrix(mean_ret_1).T
        Lambda = (G * C_inv * G.T).I * (G * C_inv * (R1 - R0) - 2 * a * c)
        w = 1 / 2 / a * C_inv * ((R1 - R0) - G.T * Lambda)

        return w

    def get_weight(self, end):
        data = self.data
        train = data.loc[:end]
        train_0 = train[train['party'] == 0]
        train_1 = train[train['party'] == 1]
        res_0, res_1, new_tickers, n = self.regression_res(train_0, train_1)

        beta_0, beta_1 = self.get_beta(res_0, res_1)
        pred_cov = self.pred_cov(train[new_tickers], 252)
        w_0 = self.optimization(beta_0, beta_1, pred_cov, 2, n)
        w_1 = self.optimization2(beta_0, beta_1, pred_cov, 2, n)
        return w_0, w_1

    def get_portfolio_miu_sigma(self, w):
        data = self.data.iloc[:, 0:30]
        ret_list = np.dot(data.dropna().values, w)
        miu = ret_list.mean()
        sigma = ret_list.var()
        return miu, sigma

    def get_portfolio_val(self, price, w):
        val = np.dot(price, w)
        return val

    def out_of_sample_test(self, t1, t2, party):
        data = self.data
        train = data.loc[:t1]
        valid = data.loc[t1:t2]
        train_0 = train[train['party'] == 0]
        train_1 = train[train['party'] == 1]
        res_0, res_1, new_tickers, n = self.regression_res(train_0, train_1)

        beta_0, beta_1 = self.get_beta(res_0, res_1)
        pred_cov = self.pred_cov(train[new_tickers], 252)
        w = self.optimization(beta_0, beta_1, pred_cov, 2, n)
        w2 = self.optimization2(beta_0, beta_1, pred_cov, 2, n)

        valid['portfolio'] = np.dot(valid[new_tickers], w).cumsum().tolist()[0]
        valid['portfolio2'] = np.dot(valid[new_tickers], w2).cumsum().tolist()[0]

        plt.figure(figsize=(8, 6))
        plt.title(t1 + ' to ' + t2 + ' ' + party)
        plt.plot(valid['portfolio'], color='red')
        plt.plot(valid['portfolio2'], color='blue')
        plt.plot(valid['SPX Index'].cumsum(), color='green')
        plt.legend(['Bet on Republican', 'Bet on Democratic', 'Market'])
        plt.show()

    def get_rf_weight(self, w, wealth, r0, a=2):
        miu, sigma = self.get_portfolio_miu_sigma(w)
        alpha = miu + 1 / 2 * sigma ** 2
        w0 = (alpha - r0) / (a * wealth * sigma ** 2)
        return w0


if __name__ == '__main__':
    # data process
    data = pd.read_csv("30 stocks log daily return(new).csv", index_col=0)
    data.index = pd.to_datetime(data.index)
    df = pd.read_csv("1974 REPUB.csv", index_col=0)
    data = pd.concat([data, np.log(df['SPX Index'].pct_change() + 1).dropna()], axis=1)
    data = data.fillna(method='ffill')

    test = OPT(data)

    # out-of-sample test
    # test.out_of_sample_test('2000-01-19', '2001-01-19', '(Republican)')
    # test.out_of_sample_test('2004-01-19', '2005-01-19', '(Republican)')
    # test.out_of_sample_test('2008-01-19', '2009-01-19', '(Democratic)')
    # test.out_of_sample_test('2012-01-21', '2013-01-21', '(Democratic)')
    # test.out_of_sample_test('2016-01-19', '2017-01-19', '(Republican)')

    w_0, w_1 = test.get_weight('2020-01-20')

    # risk free asset weight
    w0 = test.get_rf_weight(w_0, 200000, 0.00000004)
    w1 = test.get_rf_weight(w_1, 200000, 0.00000004)
    # print(w0, w1)
    # print((1 - w0) * w_0, (1 - w1) * w_1)


    price_data = pd.read_csv("stock price.csv", index_col=0)
    price = price_data.iloc[-1, :30].values
    # print(test.get_portfolio_val(price, w_0))
    # print(test.get_portfolio_val(price, w_1))

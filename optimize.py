import pandas as pd
import numpy as np
from loguru import logger
from pulp import LpVariable, LpProblem, LpMinimize


class Calculator:
    def __init__(self, m, r):
        self.M = m
        self.R = r

    def cal_y1(self, x):
        M = self.M
        if x + M <= 0: return 0
        if 0 < x + M <= 36000: return (x + M) * 3 / 100
        if 36000 < x + M <= 144000: return (x + M) * 10 / 100 - 2520
        if 144000 < x + M <= 300000: return (x + M) * 20 / 100 - 16920
        if 300000 < x + M <= 420000: return (x + M) * 25 / 100 - 31920
        if 420000 < x + M <= 660000: return (x + M) * 30 / 100 - 52920
        if 660000 < x + M <= 960000: return (x + M) * 35 / 100 - 85920
        return (x + M) * 45 / 100 - 181920

    @staticmethod
    def cal_y2(x):
        if x <= 0: return 0
        if 0 < x <= 36000: return x * 3 / 100
        if 36000 < x <= 144000: return x * 10 / 100 - 210
        if 144000 < x <= 300000: return x * 20 / 100 - 1410
        if 300000 < x <= 420000: return x * 25 / 100 - 2660
        if 420000 < x <= 660000: return x * 30 / 100 - 4410
        if 660000 < x <= 960000: return x * 35 / 100 - 7160
        return x * 45 / 100 - 15160

    @staticmethod
    def cal_y3(x):
        if x <= 0: return 0
        if 0 < x <= 36000: return x * 3 / 100
        if 36000 < x <= 144000: return x * 10 / 100 - 2520
        if 144000 < x <= 300000: return x * 20 / 100 - 16920
        if 300000 < x <= 420000: return x * 25 / 100 - 31920
        if 420000 < x <= 660000: return x * 30 / 100 - 52920
        if 660000 < x <= 960000: return x * 35 / 100 - 85920
        return x * 45 / 100 - 181920

    def cal_y4(self, x):
        return x * self.R


class Optimizer:
    def __init__(self, x, m, b, r, exit):
        self.cal = Calculator(m, r)
        self.X = x
        self.B = b
        self.M = m
        self.exit = exit
        self.prob = None

    def cond_y1(self):
        LARGE = 1e9
        cond_num = 7
        M = self.M
        b = [-M, 36000 - M, 144000 - M, 300000 - M, 420000 - M, 660000 - M, 960000 - M, LARGE - M]
        f_b = [self.cal.cal_y1(bi) for bi in b]

        w = [LpVariable(f'w1_{i}', ) for i in range(1, cond_num + 1 + 1)]
        z = [LpVariable(f'z1_{i}', cat='Binary') for i in range(1, cond_num + 1)]

        self.prob += (w[0] <= z[0])
        for i in range(1, cond_num):
            self.prob += (w[i] <= z[i - 1] + z[i])
        self.prob += (w[-1] <= z[-1])

        for w_i in w: self.prob += (w_i >= 0)

        self.prob += (sum(w) == 1)
        self.prob += (sum(z) == 1)

        x1 = np.dot(b, w)
        y1 = np.dot(f_b, w)

        self.prob += (x1 >= 0)
        return x1, y1

    def cond_y2(self):
        LARGE = 1e9
        cond_num = 7
        b = [0, 36000, 144000, 300000, 420000, 660000, 960000, LARGE]
        f_b = [self.cal.cal_y2(bi) for bi in b]

        w = [LpVariable(f'w2_{i}', ) for i in range(1, cond_num + 1 + 1)]
        z = [LpVariable(f'z2_{i}', cat='Binary') for i in range(1, cond_num + 1)]

        self.prob += (w[0] <= z[0])
        for i in range(1, cond_num):
            self.prob += (w[i] <= z[i - 1] + z[i])
        self.prob += (w[-1] <= z[-1])

        for w_i in w: self.prob += (w_i >= 0)

        self.prob += (sum(w) == 1)
        self.prob += (sum(z) == 1)
        x2 = np.dot(b, w)
        y2 = np.dot(f_b, w)

        self.prob += (x2 >= 0)
        return x2, y2

    def cond_y2_2(self, in_range: int):
        # 税负函数y2是分段函数，但非连续、单调，会破坏问题的凸性，因此只能拆出来分段汇总局部最优解，然后再取局部最优解中的最优解
        LARGE = 1e9
        b = [0, 36000, 144000, 300000, 420000, 660000, 960000, LARGE]
        range_list = [[b[i], b[i + 1]] for i in range(len(b) - 1)]
        x2 = LpVariable(f'x2')
        left, right = range_list[in_range - 1]
        if in_range == 1:
            y2 = x2 * 3 / 100
        elif in_range == 2:
            y2 = x2 * 10 / 100 - 210
        elif in_range == 3:
            y2 = x2 * 20 / 100 - 1410
        elif in_range == 4:
            y2 = x2 * 25 / 100 - 2660
        elif in_range == 5:
            y2 = x2 * 30 / 100 - 4410
        elif in_range == 6:
            y2 = x2 * 35 / 100 - 7160
        else:
            y2 = x2 * 45 / 100 - 15160
        self.prob += (left + 0.01 <= x2)
        self.prob += (x2 <= right)
        return x2, y2

    def cond_y3(self):
        if self.exit:
            LARGE = 1e9
            cond_num = 7
            b = [0, 36000, 144000, 300000, 420000, 660000, 960000, LARGE]
            f_b = [self.cal.cal_y3(bi) for bi in b]

            w = [LpVariable(f'w3_{i}', ) for i in range(1, cond_num + 1 + 1)]
            z = [LpVariable(f'z3_{i}', cat='Binary') for i in range(1, cond_num + 1)]

            self.prob += (w[0] <= z[0])
            for i in range(1, cond_num):
                self.prob += (w[i] <= z[i - 1] + z[i])
            self.prob += (w[-1] <= z[-1])

            for w_i in w: self.prob += (w_i >= 0)

            self.prob += (sum(w) == 1)
            self.prob += (sum(z) == 1)

            x3 = np.dot(b, w)
            y3 = np.dot(f_b, w)
        else:
            x3 = LpVariable(f'x3')
            y3 = 0.0 * x3
            self.prob += (x3 == 0)
        self.prob += (x3 >= 0)
        return x3, y3

    def cond_y4(self):
        x4 = LpVariable('x4')
        y4 = self.cal.cal_y4(x4)
        self.prob += (x4 >= 0)
        self.prob += (x4 <= self.B)
        return x4, y4

    def run(self):
        res_list = []
        for x2_in_range in range(0, 7):
            # 初始化问题
            self.prob = LpProblem('Bonus Problem', LpMinimize)
            x1, y1 = self.cond_y1()
            x2, y2 = self.cond_y2_2(x2_in_range + 1)
            x3, y3 = self.cond_y3()
            x4, y4 = self.cond_y4()
            self.prob += (x1 + x2 + x3 + x4 == X)
            self.prob += y1 + y2 + y3 + y4

            self.prob.solve()

            sub_df = pd.DataFrame([
                [X, x1.value(), x2.value(), x3.value(), x4.value(),
                 y1.value(), y2.value(), y3.value(), y4.value(), self.prob.objective.value()]
            ], columns=['X', 'X1工资', 'X2年终奖', 'X3离职补偿', 'X4通道', 'Y1', 'Y2', 'Y3', 'Y4', 'Y总税负']).round(2)
            res_list.append(sub_df)
        df = pd.concat(res_list)
        for col in ['X1工资', 'X2年终奖', 'X3离职补偿', 'X4通道']:
            df = df[df[col] >= 0]
        df = df.sort_values('Y总税负').reset_index(drop=True)
        return df.iloc[0]


class DataReader:
    def __init__(self):
        self.data = self.read()

    @staticmethod
    def read():
        df = pd.read_excel('/Users/zhouyuxuan/Downloads/年终奖拆分-20230110（test）.xlsx', dtype={'编号': str})
        df.rename(columns={'年终奖': '总金额'}, inplace=True)
        df['是否离职'] = df['是否离职'].apply(lambda x: True if x == '是' else False)
        return df

    def iter(self):
        for _, s in self.data.iterrows():
            num, X, M, exit, B = s['编号'], s['总金额'], s['基本薪资M'], s['是否离职'], s['通道上限（B）']
            R = 10 / 106
            yield num, X, M, B, R, exit


if __name__ == '__main__':
    dr = DataReader()
    res_list = []
    for num, X, M, B, R, exit in dr.iter():
        self = Optimizer(X, M, B, R, exit)
        res = self.run()
        res['编号'] = num
        res_list.append(res)
    df = pd.concat(res_list, axis=1).T
    df_all = pd.merge(dr.data, df, on=['编号'])
    df_all.to_excel('/Users/zhouyuxuan/Downloads/年终奖拆分-test结果.xlsx', index=False)

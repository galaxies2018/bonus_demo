import pandas as pd
from pulp import LpVariable, LpProblem, LpMinimize

a = [80.154835, 6.84317, 2.358416, 1.636269, 0.706175, 1.336279,
     0.943367, 5.017907, 0.501791, 0.501791, 0.0, 0.0]
b = [80.104729, 7.971037, 2.401435, 1.666115, 0.478219, 1.360653,
     0.698523, 3.613367, 0.765504, 0.765504, 0.095407, 0.079506]
a = [k / 100 for k in a]
b = [k / 100 for k in b]


def show_fee():
    """现有股权比例和目标股权比例"""
    df = pd.DataFrame([a, b]).T
    df.columns = ['Old', 'Target']
    df['Diff'] = df['Target'] - df['Old']
    # 单位：万元
    base_cap = 100
    fee = round(df[df['Diff'] > 0]['Diff'].sum() / 100 * base_cap * 35 * 0.2, 2)
    print(fee)


if __name__ == '__main__':
    prob = LpProblem('add cap problem', LpMinimize)
    x = [LpVariable(f'x_{i}', lowBound=0, upBound=None, cat='Continuous', e=None) for i in range(12)]
    sum_x, sum_a, sum_b = sum(x), sum(a), sum(b)
    prob += sum_x
    print(sum_x)
    for i in range(12):
        prob += (x[i] - b[i] * sum_x ==100* b[i] * sum_a - 100*a[i])

    prob.solve()
    for i in range(12):
        print(x[i].value())
    print(sum_x.value())
    # print(prob.objective.value())
    res = [x.value() for x in x]
    print(res)
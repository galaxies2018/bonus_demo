import pandas as pd
from loguru import logger
from pulp import LpVariable, LpProblem, LpMinimize

X = 1000000
M = 500000
B = 200000
R = 6 / 106
exit = True
prob = LpProblem('Bonus Problem', LpMinimize)


def cond_y1(prob):
    LARGE = 1e9
    cond_num = 7
    b = [-M, 36000 - M, 144000 - M, 300000 - M, 420000 - M, 660000 - M, 960000 - M, LARGE - M]
    f_b = [
        (b[0] + M) * 0,
        (b[1] + M) * 3 / 100,
        (b[2] + M) * 10 / 100 - 2520,
        (b[3] + M) * 20 / 100 - 16920,
        (b[4] + M) * 25 / 100 - 31920,
        (b[5] + M) * 30 / 100 - 52920,
        (b[6] + M) * 35 / 100 - 85920,
        (b[7] + M) * 45 / 100 - 181920,
    ]

    w = [LpVariable(f'w1_{i}', ) for i in range(1, cond_num + 1 + 1)]
    z = [LpVariable(f'z1_{i}', cat='Binary') for i in range(1, cond_num + 1)]

    prob += (w[0] <= z[0])
    for i in range(1, cond_num):
        prob += (w[i] <= z[i - 1] + z[i])
    prob += (w[-1] <= z[-1])

    for w_i in w: prob += (w_i >= 0)

    prob += (sum(w) == 1)
    prob += (sum(z) == 1)

    x1 = sum([b[i] * w[i] for i in range(cond_num)])
    y1 = sum([f_b[i] * w[i] for i in range(cond_num + 1)])

    prob += (x1 >= 0)
    return x1, y1, prob


def cond_y2(prob):
    LARGE = 1e9
    cond_num = 7
    b = [0, 36000, 144000, 300000, 420000, 660000, 960000, LARGE]
    f_b = [
        b[0] * 0,
        b[1] * 3 / 100,
        b[2] * 10 / 100 - 210,
        b[3] * 20 / 100 - 1410,
        b[4] * 25 / 100 - 2660,
        b[5] * 30 / 100 - 4410,
        b[6] * 35 / 100 - 7160,
        b[7] * 45 / 100 - 15160,
    ]

    w = [LpVariable(f'w2_{i}', ) for i in range(1, cond_num + 1 + 1)]
    z = [LpVariable(f'z2_{i}', cat='Binary') for i in range(1, cond_num + 1)]

    prob += (w[0] <= z[0])
    for i in range(1, cond_num):
        prob += (w[i] <= z[i - 1] + z[i])
    prob += (w[-1] <= z[-1])

    for w_i in w: prob += (w_i >= 0)

    prob += (sum(w) == 1)
    prob += (sum(z) == 1)

    x2 = sum([b[i] * w[i] for i in range(cond_num)])
    y2 = sum([f_b[i] * w[i] for i in range(cond_num + 1)])

    prob += (x2 >= 0)
    return x2, y2, prob


def cond_y3(prob, exit=False):
    if exit:
        LARGE = 1e9
        cond_num = 7
        b = [0, 36000, 144000, 300000, 420000, 660000, 960000, LARGE]
        f_b = [
            b[0] * 0,
            b[1] * 3 / 100,
            b[2] * 10 / 100 - 2520,
            b[3] * 20 / 100 - 16920,
            b[4] * 25 / 100 - 31920,
            b[5] * 30 / 100 - 52920,
            b[6] * 35 / 100 - 85920,
            b[7] * 45 / 100 - 181920,
        ]

        w = [LpVariable(f'w3_{i}', ) for i in range(1, cond_num + 1 + 1)]
        z = [LpVariable(f'z3_{i}', cat='Binary') for i in range(1, cond_num + 1)]

        prob += (w[0] <= z[0])
        for i in range(1, cond_num):
            prob += (w[i] <= z[i - 1] + z[i])
        prob += (w[-1] <= z[-1])

        for w_i in w: prob += (w_i >= 0)

        prob += (sum(w) == 1)
        prob += (sum(z) == 1)

        x3 = sum([b[i] * w[i] for i in range(cond_num)])
        y3 = sum([f_b[i] * w[i] for i in range(cond_num + 1)])
    else:
        x3 = LpVariable(f'x3')
        y3 = 0.0 * x3
        prob += (x3 == 0)
    prob += (x3 >= 0)
    return x3, y3, prob


def cond_y4(prob):
    x4 = LpVariable('x4')
    y4 = x4 * R
    prob += (x4 >= 0)
    prob += (x4 <= B)
    return x4, y4, prob


x1, y1, prob = cond_y1(prob)
x2, y2, prob = cond_y2(prob)
x3, y3, prob = cond_y3(prob, exit=exit)
x4, y4, prob = cond_y4(prob)
prob += (x1 + x2 + x3 + x4 == X)
prob += y1 + y2 + y3 + y4

prob.solve()

df = pd.DataFrame([
    [x1.value(), x2.value(), x3.value(), x4.value(), X],
    [y1.value(), y2.value(), y3.value(), y4.value(), prob.objective.value()]
], columns=['工资', '年终奖', '离职补偿', '通道', '合计'], index=['金额', '税负']).T.round(2)

logger.info(f'输入参数：\n'
            f'员工应发放的总金额 {X}\n'
            f'员工基本工资薪金 {M}\n'
            f'是否离职 {exit}\n'
            f'通道上限 {B}\n'
            f'通道税率 R=6/106')

logger.info('\n' + df.to_string())

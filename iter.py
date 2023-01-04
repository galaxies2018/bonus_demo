import pandas as pd


def y1(x):
    if x <= 0: return 0
    if 0 < x <= 36000: return x * 3 / 100
    if 36000 < x <= 144000: return x * 10 / 100 - 2520
    if 144000 < x <= 300000: return x * 20 / 100 - 16920
    if 300000 < x <= 420000: return x * 25 / 100 - 31920
    if 420000 < x <= 660000: return x * 30 / 100 - 52920
    if 660000 < x <= 960000: return x * 35 / 100 - 85920
    return x * 45 / 100 - 181920


def y2(x):
    if x <= 0: return 0
    if 0 < x <= 36000: return x * 3 / 100
    if 36000 < x <= 144000: return x * 10 / 100 - 210
    if 144000 < x <= 300000: return x * 20 / 100 - 1410
    if 300000 < x <= 420000: return x * 25 / 100 - 2660
    if 420000 < x <= 660000: return x * 30 / 100 - 4410
    if 660000 < x <= 960000: return x * 35 / 100 - 7160
    return x * 45 / 100 - 15160


y3 = y1


def y4(x):
    R = 6 / 106
    return x * R


X = 1000000
M = 500000
B = 200000

x3 = 0
x4 = 200000
x1 = 430000
x2 = X - x3 - x4 - x1
y = y1(x1 + M) + y2(x2) + y3(x3) + y4(x4)

df = pd.DataFrame([
    [x1, x2, x3, x4, X],
    [y1(x1 + M), y2(x2), y3(x3), y4(x4), y]
], columns=['工资', '年终奖', '离职补偿', '通道', '合计'], index=['金额', '税负']).T.round(2)

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help='reward log file name')
parser.add_argument('--x_axis', '-x', default='episode', type=str,
                    help='X_axis Value : episode or cycle')


args = parser.parse_args()

df = pd.read_csv(args.log_file)

if(args.x_axis == "episode"):
    x = df.columns[2]
elif(args.x_axis == "cycle"):
    x = df.columns[0]
else:
    print u"正しいX-axisのParserを指定してください"
    import sys
    sys.exit()


y = df.columns[1]
ax = df.plot(kind='scatter', x=x, y=y)

df[y] = pd.rolling_mean(df[y], window=20)
df.plot(kind='line', x=x, y=y, ax=ax)
plt.show()

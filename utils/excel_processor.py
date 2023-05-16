import pandas as pd
import sklearn
import matplotlib.pyplot as plt

path = '../logs/FB15k-237-betae/1p.2p.3p.2i.3i.ip.pi.2u.up/vec/g-24.0/2022.03.25-19_48_14/pred.xlsx'

df = pd.read_excel(path)
acc = df['top10_type_acc'].tolist()

sub_df = df[df['top10_type_acc'] == 0]

# plt.hist(acc,bins=11)
# plt.show()


# datas = []
# for i in range(0, 10 + 1):
#     sub_df = df[df['top10_type_acc'] == i]
#     datas.append(sub_df['mrr'].tolist())
#     plt.hist(datas[-1], range=(0,1), bins=10)
#     plt.title(f'top10 type accuracy: {i}')
#     plt.show()
    # print(i, len(sub_df),sum(datas[-1]) / len(sub_df))

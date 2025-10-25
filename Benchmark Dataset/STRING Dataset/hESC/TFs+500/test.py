import pandas as pd
net_df = pd.read_csv("BL--network.csv")
net = set(net_df.Gene1) | set(net_df.Gene2)
print(len(set(net_df.Gene1)), len(set(net_df.Gene2)), len(net))
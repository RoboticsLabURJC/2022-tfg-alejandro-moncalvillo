import time
import pandas as pd
import matplotlib.pyplot as plt

df_1 = pd.read_csv("./simple_holonomic/data.csv")
print(df_1.groupby('Type').size())
"""
figure, axis = plt.subplots(2, 2)

figure.tight_layout()
df_1 = pd.read_csv("./simple_holonomic/data.csv")
print(df_1.groupby('Type').size())
print(df_1.groupby('Type').mean())
df_1.groupby('Type').size().plot(kind='pie', ax= axis[0,0])
axis[0, 0].set_title("Simple circuit")

df_2 = pd.read_csv("./montmelo_holonomic/data.csv")
print(df_2.groupby('Type').size())
print(df_2.groupby('Type').mean())
df_2.groupby('Type').size().plot(kind='pie', ax= axis[0,1])
axis[0, 1].set_title("Montmelo")

df_3 = pd.read_csv("./montreal_holonomic/data.csv")
print(df_3.groupby('Type').size())
print(df_3.groupby('Type').mean())
df_3.groupby('Type').size().plot(kind='pie', ax= axis[1,0])
axis[1, 0].set_title("Montreal")

df_4 = pd.read_csv("./nurburgring_holonomic/data.csv")
print(df_4.groupby('Type').size())
print(df_4.groupby('Type').mean())
df_4.groupby('Type').size().plot(kind='pie',ax= axis[1,1])
axis[1, 1].set_title("Nurburgring")
"""


plt.show()

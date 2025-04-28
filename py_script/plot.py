import matplotlib.pyplot as plt
import numpy as np

# CSVファイルから損失関数を読み込む
loss1 = []
with open("loss_history4.csv", "r") as f:
    for line in f:
        loss1.append(float(line.strip()))

# プロットする
plt.plot(range(1, len(loss1) + 1), loss1,marker = "o")
plt.xlabel("epoc")
plt.ylabel("loss_average")
plt.title("loss_average per epoc")
plt.grid(True)
plt.tight_layout()
plt.show()

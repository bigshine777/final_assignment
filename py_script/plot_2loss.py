import matplotlib.pyplot as plt
import numpy as np

# CSVファイルから損失関数を読み込む
loss1 = []
loss2 = []
with open("loss_history/loss_history_sgd.csv", "r") as f:
    for line in f:
        loss1.append(float(line.strip()))

with open("loss_history/loss_history_adam.csv", "r") as f:
    for line in f:
        loss2.append(float(line.strip()))

# プロットする
plt.plot(range(1, len(loss1) + 1), loss1, color="blue",label = "SGD")
plt.plot(range(1, len(loss2) + 1), loss2, color="red",label = "Adam")
plt.xlabel("epoc")
plt.ylabel("loss_average")
plt.title("loss_average per epoc")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

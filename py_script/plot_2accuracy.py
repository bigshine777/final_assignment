import matplotlib.pyplot as plt
import numpy as np

# CSVファイルから損失関数を読み込む
accuracy1 = []
accuracy2 = []
with open("accuracy_history/accuracy_history_train_increased.csv", "r") as f:
    for line in f:
        accuracy1.append(float(line.strip()))

with open("accuracy_history/accuracy_history_test_increased.csv", "r") as f:
    for line in f:
        accuracy2.append(float(line.strip()))

# プロットする
plt.plot(range(1, len(accuracy1) + 1), (accuracy1), color="blue",label = "train")
plt.plot(range(1, len(accuracy2) + 1), (accuracy2), color="red",label = "test")
plt.xlabel("epoc")
plt.ylabel("accuracy average")
plt.title("accuracy average per epoc")
plt.legend()
plt.grid(True)
# yの最大最小値を設定
plt.ylim(86.4, 100.4) 
plt.show()

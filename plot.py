import matplotlib.pyplot as plt

# CSVファイルから損失値を読み込む
losses = []
with open('loss_history.csv', 'r') as f:
    for line in f:
        losses.append(float(line.strip()))

# プロット
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.xlabel('Minibatch')
plt.ylabel('Loss')
plt.title('Training Loss Over minibatch')
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# part1
# 定义函数E，参数k和q
def E(k, q):
    return 1 + 1 / k - q ** k  # 返回数学期望的计算公式


def E2(k, q, q2):
    return (1 / k) * (q ** k) + (1 + (1 / k)) * (1 - q2) * (1 - q ** k) + ((1 / k) + 2) * (q2 * (1 - q ** k))


healthyarray = [0.8, 0.9, 0.99]  # 健康人占比
non_infect = [0.2,0.5,0.7]
karray = np.linspace(0, 20, 1000)  # 取一定范围k值

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# 遍历健康人占比的的取值范围命名为q，计算E的值，并绘制图像
for n in non_infect:
    for q in healthyarray:
        y = E2(karray, q, n)
        ax.plot(karray[30:], y[30:],n, label=f'q = {round(1 - q, 4)}')  # 作当前q值下的图线
        minexcept = int(np.argmin(y[1:]))
        print(minexcept)
        ax.scatter(karray[minexcept],np.nanmin(y),n, s=10, c='r')  # 找出数学期望最低值和最低值索引对应的k，打点
        ax.text(karray[minexcept],np.nanmin(y),n, c='b',s=str(karray[minexcept]))  # 给出点值

# 设置x轴和y轴的标签
ax.set_xlabel('k')
ax.set_ylabel('E(X)')
ax.set_zlabel('q2')
ax.set_ylim(0, 5)  # 设置y轴范围为[0, 15]
ax.legend()  # 添加图例
ax.set_title('Math expecation in different k and different q2')
plt.show()

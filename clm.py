
# Authored by Jack Wu
from numpy import *  # 可以直接使用npy内所有方法
from numpy.linalg import det  # det:线性代数中求解行列式的方法


class get_solution:  # 在每一列插入常数
    def __init__(self, deter_array, const):  # 初始化类需要传入:①线性方程组对应行列式deter_array ②线性方程组对应的常数const
        self.deter_array = deter_array
        self.const = const
        self.x_result=[]
        self.clm_array = []  # 在每一列插入常数后的接收表

    def get_clm_array(self):  # 调用此方法可以每列插入常数,填入接收表clm_array
        for chg_col in range(len(self.deter_array[0])):  # 此层遍历每一列,列数取样于第一列 chg_col为列索引
            clm_array_tmp = []  # 临时表:这一列全部更改为常数后临时存于此,每遍历到新一列清空

            for i in range(len(self.deter_array)):  # 此层遍历行,获得行索引i
                this_row = [] # 本行数据临时存储于这,每遍历一次行清空

                for j in range(len(self.deter_array[i])):  # 此层遍历列,获得列索引j

                    if j != chg_col:  # 本列不属于更改列
                        this_row.append(self.deter_array[i][j])  # 本行临时表为原来行列的数据(Xij)

                    elif j == chg_col:  # 本列是修改列
                        this_row.append(self.const[i])  # 修改为给出对应列的常数

                clm_array_tmp.append(this_row)  # 将每行加入临时行列式

            self.clm_array.append(clm_array_tmp)  # 将临时行列式组装进行列式组

        for i in range(len(self.clm_array)):
            Dj = array(self.clm_array[i])
            result_x=det(Dj) / det(array(self.deter_array))
            self.x_result.append(result_x)

        return self.x_result

if __name__ == '__main__':
    deter_array = [[1, 2, 3, 4, 5],[16, 17, 18, 19, 6],[15, 24, 25, 20, 7],[14, 23, 22, 21, 8],[13, 12, 11, 10, 9]]
    b = [0,0,2,-2,2]
    D = array(deter_array)
    result = det(D)

    clm_array = get_solution(deter_array, b).get_clm_array()

    for i in range(len(clm_array)):  # 解开行列式组
        Dj = array(clm_array[i])
        result_x = det(Dj) / result
        print("其中,X{}为:{}".format(i + 1, result_x))

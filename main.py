#  Authored Jack Wu
#  软工5班吴宇杰期末作业
import numpy as np
import sympy
from clm import get_solution as clm_solution



class LinerAlgebraCalculator:
    def __init__(self):
        self.welcome = '欢迎使用马鹿牌线性代数计算器\n请选择功能（按q退出）'
        self.service_map_det = {'name': '行列式',
                                'servicelst': {
                                               '0': {'name': '余子式计算（Mij）'},
                                               '1': {'name': '代数余子式计算（Aij）'},
                                               '2': {'name': '计算行列式的值'},
                                               '3': {'name': '克莱姆法则求解线性方程组未知量'}
                                                }
                                }
        self.service_map_matrix = {'name': '矩阵',
                                   'servicelst': {'0': {'name': '两矩阵相乘'},
                                                  '1': {'name': '矩阵k次方'},
                                                  '2': {'name': '矩阵的逆'},
                                                  '3': {'name': '求伴随矩阵'},
                                                  '4': {'name': '矩阵的方程'},
                                                  '5': {'name': '矩阵的秩'},
                                                  }
                                   }

        self.service_map = {'0':self.service_map_det,'1':self.service_map_matrix}
        self.matmen = None

    def get_const(self):
        const=[]
        while len(const) != self.j:
            const = list(map(int, input('输入线性方程组常数项,有多少列输多少,使用空格隔开').split()))
            if len(const) != self.j:
                print('请正确输入!')
        return const
    def cofactor_M(self,A,i,j):  # 余子式
        Mij = np.delete(np.delete(A, i-1, axis=0), j-1, axis=1)
        return Mij

    def cofactor(self,A,i,j):
        Mij = self.cofactor_M(A,i,j)
        Cij = ((-1) ** (i + j)) * np.linalg.det(Mij)
        return Cij

    def get_det(self):
        deter_array = []
        while True:
            Lij = list(map(int, input('行列式有几行几列? 如第2行2列输入:2 2').split()))
            if len(Lij)!=2:
                print('请正确输入!')
                continue
            else:
                break

        self.i = Lij[0]
        self.j = Lij[1]

        for i in range(self.i):
            c = list(map(int,input('请输入第{}行的数字,用空格隔开!>>'.format(i+1)).split()))
            while len(c) != self.j:
                print('与列数不相等,请重新输入本行')
                c = list(map(int, input('请输入第{}行的数字,用空格隔开!>>'.format(i+1)).split()))
            deter_array.append(c)

        return np.array(deter_array)

    def get_mat_process(self,mat_array):
        new_mat_array=[]
        for array in mat_array:
            thisline=[]
            for a in array:
                try:
                    thisline.append(int(a))
                except:
                    thisline.append(sympy.symbols(a))
            new_mat_array.append(thisline)

        return new_mat_array


    def get_mat(self):
        mat_array = []
        while True:
            Lij = list(map(int, input('矩阵有几行几列? 如第2行2列输入:2 2').split()))
            if len(Lij)!=2:
                print('请正确输入!')
                continue
            else:
                break

        self.i = Lij[0]
        self.j = Lij[1]

        for i in range(self.i):
            c = list(input('请输入第{}行的数字,用空格隔开!>>'.format(i + 1)).split())
            while len(c) != self.j:
                print('与列数不相等,请重新输入本行')
                c = list(input('请输入第{}行的数字,用空格隔开!>>'.format(i + 1)).split())
            mat_array.append(c)

        mat_array = self.get_mat_process(mat_array)
        print(mat_array)

        return sympy.Matrix(mat_array)

    def mat_mulit(self,A,B):
        return A*B

    def mat_mulit_times(self,A,k):
        try:
            k = int(k)
        except:
            k=sympy.symbols(k)
        return A**k

    def mat_inv(self,A):
        return A.inv()

    def mat_mulit_memo(self):
        leftorright = None
        while leftorright not in ['l', 'r']:
            leftorright = input('记录矩阵放左边还是右边？（l/r）'.format(self.matmen))
            if leftorright == 'l':
                A=self.matmen
                B = self.get_mat()
                self.matmen = self.mat_mulit(A,B)
            elif leftorright == 'r':
                A = self.get_mat()
                B = self.matmen
                self.matmen = self.mat_mulit(A,B)
            else:
                print('请正确输入')

    def det_services(self):
        while True:
            print('你位于行列式服务菜单,输入q返回')
            for item in self.service_map['0']['servicelst']:
                print("使用{}功能请输入 {} ".format(self.service_map_det['servicelst'][item]['name'], item))

            i = input('>>')
            if i == '0':
                detarray=self.get_det()
                Lij = list(map(int,input('你希望得到第几行第几列的余子式? 如第3行第3列输入:3 3').split()))
                cofM = self.cofactor_M(detarray,Lij[0],Lij[1])
                print("第{}行{}列的余子式是:{}".format(self.i,self.j,cofM))
                print('余子式值{}'.format(round(np.linalg.det(cofM),1)))
                print('计算完毕\n')
            elif i == '1':
                detarray=self.get_det()
                Lij = list(map(int,input('你希望得到第几行第几列的余子式? 如第3行第3列输入:3 3').split()))
                cof = self.cofactor(detarray,Lij[0],Lij[1])
                print("第{}行{}列的代数余子式是:{}".format(Lij[0],Lij[1],cof))
                print('代数余子式值{}'.format(round(np.linalg.det(cof),1)))
                print('计算完毕\n')
            elif i == '2':
                detarray=self.get_det()
                print("行列式的值为{}".format(round(np.linalg.det(detarray),1)))
                print('计算完毕\n')
            elif i == '3':
                print('请把线性方程组系数变成行列式再输入')
                detarray=self.get_det()
                result = np.linalg.det(np.array(detarray))
                const=self.get_const()
                x_lst=clm_solution(detarray,const).get_clm_array()
                for i in range(len(x_lst)):  # 解开行列式组
                    print("X{}为:{}".format(i + 1, round(x_lst[i],1)))
                    print('计算完毕\n')

            elif i == 'q':
                break

    def mat_company(self,A):
        return np.linalg.det(A)*A.inv()

    def matrix_services(self):
        while True:
            print('你位于矩阵服务菜单,输入q返回,含有未知数可直接输入')
            for item in self.service_map['1']['servicelst']:
                print("使用{}功能请输入 {} ".format(self.service_map_matrix['servicelst'][item]['name'], item))
            i = input('>>')

            if i == '0':
                if self.matmen != None:
                    yesorno = input('检测到之前的计算记录{}，是否继续用它相乘？(y/n)'.format(self.matmen))
                    if yesorno == 'y':
                        self.mat_mulit_memo()
                        print('相乘结果：'.format(self.matmen))
                        print('计算完毕\n')
                        continue
                print("请注意先后顺序")
                print('左',end='')
                A = self.get_mat()
                print('右',end='')
                B = self.get_mat()
                self.matmen=self.mat_mulit(A,B)
                print('计算结果为{}'.format(self.matmen))
                print('计算完毕\n')
            elif i == '1':
                k = input('请输入幂K')
                A = self.get_mat()
                self.matmen = self.mat_mulit_times(A,k)
                print('相乘结果:{}'.format(self.matmen))
                print('计算完毕\n')
            elif i == '2':
                if self.i != self.j:
                    print('不是方阵！\n')
                    continue
                print('逆矩阵是:{}'.format(self.mat_inv(self.get_mat())))
                print('计算完毕\n')
            elif i == '3':
                print('伴随矩阵是:{}'.format(self.mat_company(self.get_mat())))
                print('计算完毕\n')
            elif i == '4':
                while True:
                    leftorright = input("矩阵未知数X在左还是在右？（l/r）")
                    if leftorright == 'l':
                        print('XA=B中的A是？',end='')
                        A = self.get_mat()
                        print('XA=B中的B是？',end='')
                        B = self.get_mat()
                        self.matmen = self.mat_mulit(B,A.inv())
                        print('矩阵方程X是:{}'.format(self.matmen))
                        print('计算完毕\n')
                        break
                    elif leftorright == 'r':
                        print('AX=B中的A是？', end='')
                        A = self.get_mat()
                        print('AX=B中的B是？',end='')
                        B = self.get_mat()
                        self.matmen = self.mat_mulit(A.inv(),B)
                        print('矩阵方程X是:{}'.format(self.matmen))
                        print('计算完毕\n')
                        break

            elif i == '5':
                rank=self.get_mat().rank()
                print('矩阵的秩是:{}'.format(rank))
                print('计算完毕\n')
            elif i == 'q':
                break

    def mainloop(self):
        print(self.welcome)
        while True:
            for item in self.service_map:
                print("欲进行{}计算请输入 {} ".format(self.service_map[item]['name'], item))

            i = input('>>')

            if i == '0':
                self.det_services()
            elif i == '1':
                self.matrix_services()
            elif i == 'q':
                print('再见')
                exit(0)
            else:
                print('请正确输入!')


app = LinerAlgebraCalculator()
app.mainloop()
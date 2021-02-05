# Max z =  18*x1 + 12.5*x2
#
# s.t. x1 + x2 +x3 = 20
#
# x1 + x4 = 12
#
# x2 + x5 = 16
#
# x1,x2,x3,x4,x5 >= 0
#
# 运行下面脚本(单纯形法的Python实现)

# https://www.cnblogs.com/stenci/p/11910125.html

import numpy as np

# 实体类　Table
# 控制类　Simplex

class Table:

    def __init__(self):

        pass

    def set_para(self,A,b,c,base,z0):

        """

        输入LP必须已经化为标准形式

        """

        self.A=A

        self.b=b

        self.c=c

        self.z0=z0

        self.base=base

        self.m,self.n=self.A.shape

    def build(self):

        self.table=np.zeros((self.m+1,self.n+1))

        self.table[:-1,:1]=self.b.T

        self.table[-1 ,0]=self.z0

        self.table[:-1,1:]=self.A

        self.table[-1, 1:]=c

        self.baseVar=base

    def is_best(self):

        for sigma_index in range(self.n):

            if sigma_index not in self.baseVar:

                sigma=self.table[-1,1+sigma_index]

                if sigma>0:

                    return False

        return True

    def is_no_solution(self):

        for sigma_index in range(self.n):

            if sigma_index not in self.baseVar:

                sigma=self.table[-1,1+sigma_index]

                if sigma>0:

                    no_solution_flag=True

                    for a in self.table[:-1,1+sigma_index]:

                        if a>0:

                            no_solution_flag=False

                    if no_solution_flag==True:

                        return True

        return False

    def get_inVar(self):

        max_sigma=0

        inVar=None

        for sigma_index in range(self.n):

            if sigma_index not in self.baseVar:

                sigma=self.table[-1,1+sigma_index]

                if sigma>max_sigma:

                    max_sigma=sigma

                    inVar=sigma_index

        return inVar


    def get_outVar(self,inVar):

        rates=[]

        for nobaseVar in range(self.m):

            a=self.table[nobaseVar,inVar+1]

            b=self.table[nobaseVar,     0 ]

            if a>0:

                rate=b/a

                rates.append((rate,nobaseVar))

        return min(rates)[1]

    def in_out(self,inVar,outVar):

        a=self.table[outVar,inVar+1]

        self.table[outVar,:]/=a

        for i in range(self.m+1):

            if i != outVar:

                self.table[i,:]-=self.table[outVar,:]*self.table[i,inVar+1]

        self.baseVar[outVar]=inVar


    def show(self):

        print ('基变量/取值：',self.baseVar,end='/')

        print (self.table[:-1,0])

        print ("单纯形表")

        for i in range(self.m+1):

            for j in range(self.n+1):

                print ('%6.2f'%self.table[i,j],end=' ')

            print ()

        print ()

class Simplex:

    def __init__(self):

        self.table=Table()

        # 0 正常，尚未得到最优解，继续迭代

        # 1 无解，无界解

        # 2 达到最优解

        self.status=0

        self.inVar=None

        self.outVar=None

    def set_para(self,A,b,c,base,z0=0):

        self.table.set_para(A,b,c,base,z0)

    def output_result(self):

        self._main()

        if self.status==1:

            print("此问题无界")

        elif self.status==2:

            print("此问题有一个最优解")

        elif self.status==3:

            print("此问题有无穷多个最优解")

    def _main(self):

        self._build_table()

        while 1:

            self.table.show()

            if self._is_best() or self._is_no_solution():

                return

            self._get_inVar()

            self._get_outVar()

            self._in_out()

    def _build_table(self):

        self.table.build()

    def _is_best(self):

        if self.table.is_best():

            self.status=2

            return True

        return False

    def _is_no_solution(self):

        if self.table.is_no_solution():

            self.status=1

            return True

        return False


    def _get_inVar(self):

        self.inVar=self.table.get_inVar()


    def _get_outVar(self):

        self.outVar=self.table.get_outVar(self.inVar)


    def _in_out(self):

        self.table.in_out(self.inVar,self.outVar)



if __name__=="__main__":

    s=Simplex()

    A=np.matrix([[1,1,1,0,0],

                 [ 1, 0,0,1,0],

                 [ 0, 1,0,0,1]])

    b=np.matrix([[20,12,16]])

    c=np.matrix([[18,12.5,0,0,0]])

    base=[2,3,4]

    s.set_para(A,b,c,base)

s.output_result()
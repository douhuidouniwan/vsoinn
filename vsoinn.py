from soinn import Soinn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum)
from sympy import symbols, diff



# VISUALISATION
from updateWeight import UpdateWeight



def plot_low_dimensional_embedding(X_low, name, color):
    plt.figure()
    ax = plt.subplot(111)
    # plot the MPG value color-coded
    # sc = ax.scatter(X_low[:, 0], X_low[:, 1], s=20,c=color, cmap=plt.cm.gist_ncar,linewidths=12,marker='d')
    ax.scatter(X_low[:, 0], X_low[:, 1], s=20,c=color, alpha=0.5, cmap=plt.cm.rainbow)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    # plt.title("{}".format(name))

    # imgName = "curve_" + name + ".png";
    # plt.savefig('D:\image\\' + imgName ,dpi=200, bbox_inches='tight')


def plot_low_dimensional_embedding_circle(X_low, number, name, color):

    total = sum(number)
    size = np.array([], dtype=np.int32)
    for i in range(len(number)):
        n = size.shape[0]
        size.resize(n + 1, 1, refcheck=False)
        size[-1, :] = number[i]/total * 100 * 500

    plt.figure()
    ax = plt.subplot(111)
    # sc = ax.scatter(X_low[:, 0], X_low[:, 1], s=size,c=color, cmap=plt.cm.rainbow,linewidths=12,marker='d')
    ax.scatter(X_low[:, 0], X_low[:, 1], s=size,c=color, alpha=0.5, cmap=plt.cm.rainbow)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.title("{}".format(name))



def dis(x1,x2):
    n = len(x1)
    s = 0.0
    for i in range(n):
        s = s + (x1[i] - x2[i])*(x1[i] - x2[i])
    s=s ** 0.5
    return s

def rsd(arr_n,arr_2):
    rsd = 0
    d=np.array([], dtype=np.float64)
    # 计算高维中最近的1个点
    for i in range(len(arr_n)):
        x = arr_n[i]
        mindis_n = 10000
        dis_n = 100
        dis_2 = 100
        for j in range(len(arr_n)):
            y = arr_n[j]
            dis_n = dis(x,y)
            if i!=j and dis_n==0:
                continue
            if dis_n < mindis_n and i!=j:
                mindis_n = dis_n
                dis_2 = dis(arr_2[i],arr_2[j])
        n_d = d.shape[0]
        d.resize(n_d + 1, 1, refcheck=False)
        d[-1, :] = dis_n/dis_2
    u = np.mean(d)
    std = np.std(d,ddof=1)
    rsd = std/u
    return rsd

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return np.sqrt(dot(x, x.T))

def SOINN_OUTPUT(X,delete_node_period,init_node_num):
    # 2 initialize SOINN
    s = Soinn(delete_node_period=delete_node_period, init_node_num=init_node_num)
    s.fit(X)

    # 3 soinn output
    s_nodes = s.nodes
    s_edges = s.adjacent_mat.keys()
    s_count = s_nodes.shape[0]
    s_nodeMark = s.nodeMark
    s_baseWinner = np.array([], dtype=np.float64)
    for i in range(s_count):
        if(s.winning_times[i]>2):
            n_base = s_baseWinner.shape[0]
            s_baseWinner.resize(n_base + 1, s_nodes.shape[1], refcheck=False)
            s_baseWinner[-1, :] = s_nodes[i]
    return s_nodes,s_nodeMark

def getDataLocation2D(baseNodes, dataLocationIndex):
    allNum = len(dataLocationIndex)
    dataLocation = np.zeros((allNum, 2))
    for i in range(allNum):
        dataLocation[i] = baseNodes[dataLocationIndex[i]]
    return dataLocation


def find_nearest_nodes(data, num: int, signal: np.ndarray):
    n = data.shape[0]
    indexes = [0] * num
    sq_dists = [0.0] * num
    D = np.sum((data - np.array([signal] * n))**2, 1)
    for i in range(num):
        indexes[i] = np.nanargmin(D)
        sq_dists[i] = D[indexes[i]]
        D[indexes[i]] = float('nan')
    return indexes, sq_dists


def eucliDist(x1,x2,n):
    s = 0.0
    for i in range(n):
        s = s + (x1[i] - x2[i])*(x1[i] - x2[i])
    s=s ** 0.5
    return s

def eucliDist2D(x1,x2_x,x2_y):
    s = 0.0
    s = s + (x1[0] - x2_x)**2
    s = s + (x1[1] - x2_y)**2
    s=s ** 0.5
    return s

def S_1(X,input2D,winner,winner_2D_x,winner_2D_y):
    sum = 0
    for i in range(len(X)):
        signal = X[i]
        signal_2D = input2D[i]
        sum = sum + (eucliDist(winner, signal,len(signal)) - eucliDist2D(signal_2D,winner_2D_x,winner_2D_y)) ** 2
    return sum


def update_winner_2d_location(X,input2D,winner,winner_2D):

    winner_2D_x,winner_2D_y = symbols('winner_2D_x winner_2D_y', real=True)
    up_x = diff(S_1(X,input2D,winner,winner_2D_x,winner_2D_y), winner_2D_x).subs({winner_2D_x:winner_2D[0], winner_2D_y:winner_2D[1]})
    up_y = diff(S_1(X,input2D,winner,winner_2D_x,winner_2D_y), winner_2D_y).subs({winner_2D_x:winner_2D[0], winner_2D_y:winner_2D[1]})
    winner_2D[0] = winner_2D[0] - 0.5*up_x
    winner_2D[1] = winner_2D[1] - 0.5*up_y
    return winner_2D



# 1 input
n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)

# **----v-soinn only
# 1
soinn_data,s_nodeMark = SOINN_OUTPUT(X,delete_node_period=500,init_node_num=50)

random_generator = random.RandomState(None)
nodes2D = random_generator.rand(len(soinn_data), 2)
for i in range(len(soinn_data)):
    # for j in range(1):
        # normalization
    norm = fast_norm(nodes2D[i])
    nodes2D[i] = nodes2D[i] / norm

random_generator = random.RandomState(None)
input2D = random_generator.rand(len(X), 2)
for i in range(len(X)):
    # for j in range(1):
        # normalization
    norm = fast_norm(input2D[i])
    input2D[i] = input2D[i] / norm

for iter in range(500):
    for i in range(len(X)):
        n = len(X)
        winner = X[i]
        winner_2D = input2D[i]
        winner_2D_0 = update_winner_2d_location(X,input2D,winner,winner_2D)

baseNodes = nodes2D



# # 2
# # quality
# rsd_vsoinn = rsd(soinn_data,baseNodes)
# print("vsoinn:{:.3f}s".format(rsd_vsoinn))

# 3
# project all nodes
p = UpdateWeight(baseNodes=baseNodes,basevectors=soinn_data)
p.update(X)
dataLocationIndex = p.nodeLocationIndex
dataLocation = getDataLocation2D(baseNodes, dataLocationIndex)
winning_times = p.winning_times

# 4
# show
color_ture = np.array([], dtype=np.float64)
for cnt, xx in enumerate(baseNodes):
    cnt_true = s_nodeMark[cnt][0]
    n = color_ture.shape[0]
    color_ture.resize(n + 1, refcheck=False)
    color_ture[-1] = color[cnt_true]
plot_low_dimensional_embedding_circle(baseNodes, winning_times, "v-soinn", color_ture)


plt.show()

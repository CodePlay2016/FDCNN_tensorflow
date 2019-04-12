import os, pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


filepath = './checkpoint/cvgg19/2019-04-12_173756/curvelist.pkl'

# mpl.rcParams['font.sans-serif'] = 'SimHei' # Chinese font
# mpl.rcParams['font.family'] = 'sans-serif'

mpl.rcParams["font.family"] = "Times New Roman"# English font
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['axes.unicode_minus'] = False

def do_average(curve, window_size):
    pass

def cal_statistics(curve, last_n):
    curve = np.array(curve[-last_n:])
    average = np.mean(curve)
    std_dev = np.std(curve)
    max_v = np.max(curve)
    min_v = np.min(curve)

def main():
    with open(filepath, 'rb') as f:
        curve = pickle.load(f)

    test_curve = curve[3]
    test_curve = [acc*100 for acc in test_curve]
    valid_curve = curve[1]
    valid_curve = [acc*100 for acc in valid_curve]
    delta = [valid_curve[i]-test_curve[i] for i in range(len(test_curve))]
    # plt.text(10, 105, "后1000次迭代平均验证准确率: %.2f %%"%(sum(test_curve[-1000:])/1000*100))
    print(sum(valid_curve[-1000:])/1000)
    print(max(valid_curve))
    print(min(valid_curve))
    plt.plot(test_curve, lw=1, label='target domain validation')
    plt.plot(valid_curve, lw=1, label='source domain validation')
    plt.xlim(0, len(test_curve))
    plt.ylim(0, 105)
    plt.xlabel('number of epoch')
    plt.ylabel('validation accuracy/%')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

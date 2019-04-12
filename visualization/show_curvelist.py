import os, pickle
import matplotlib.pyplot as plt

filepath = './checkpoint/cvgg19/2019-04-12_173756/curvelist.pkl'

def do_average(curve, window_size):
    pass

def main():
    with open(filepath, 'rb') as f:
        curve = pickle.load(f)

    test_curve = curve[3]
    plt.text(10, 1, '后1000次迭代平均验证准确率: %d'%(sum(test_curve[-1000:])/1000))
    print(sum(test_curve[-1000:])/1000)
    print(max(test_curve))
    plt.plot(test_curve)
    plt.xlim(0, len(test_curve))
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    main()
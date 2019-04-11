import os, pickle
import matplotlib.pyplot as plt

filepath = './checkpoint/cvgg19_2019-04-11_155952/curvelist.pkl'

def do_average(curve, window_size):
    pass

def main():
    with open(filepath, 'rb') as f:
        curve = pickle.load(f)

    test_curve = curve[3]
    print(sum(test_curve[-1000:])/1000)
    print(max(test_curve))
    plt.plot(test_curve)
    plt.show()

if __name__ == "__main__":
    main()
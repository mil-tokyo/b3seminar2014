import numpy as np
import sys
from nn import train, test

def main():
    mode = sys.argv[1]
    print "mode = ", mode
    train_data = np.array([[0, 0], [0, 1], [1,0], [1,1]])
    train_label = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    wHid, wOut = train(train_data, train_label, mode)
    test_data = np.array([[0,0], [0,1], [1,0], [1,1]])
    test(test_data, wHid, wOut)

if __name__ == "__main__":
    main()

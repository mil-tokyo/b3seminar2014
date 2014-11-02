import numpy as np
import pprint

def train(train_data, train_label, mode, train_param={"iteration":10000, "eta":0.05, "hidden_num":6}):
    """ calculate weights of neural network
    Input X:samples(data_num*data_dim) t:labels(data_num*label_num),
    mode:excercise mode("1" or "2"), eta: learning ratio, iteration:iteration of training
    Output wHid: weight of Hideen layer, wOut: weight of Output layer
    """
    iteration, eta, hidden_num = train_param["iteration"], train_param["eta"], train_param["hidden_num"]
    
    data_num, data_dim = train_data.shape
    label_num, label_dim= train_label.shape
    train_data = np.hstack((np.ones((data_num, 1)), train_data))
    
    
    if mode == "1":
        """excercise1: just load pre-trained weight"""
        wHid = np.load("./data/wHid.npy")
        wOut = np.load("./data/wOut.npy")
    
    elif mode == "2":
        """excercise2: calculate network weight using back propagation"""

        """initialize network weights"""
        wHid = np.random.uniform(-1, 1, size=(hidden_num, data_dim+1)) 
        wOut = np.random.uniform(-1, 1, size = (label_dim, hidden_num))
        
        for it in xrange(iteration):
            if it % 1000 == 0: print "iteration {}/{}".format(it,iteration)
                
            for i in xrange(data_num):
                """fetch data"""
                data = train_data[i, :]
                label = train_label[i, :]
                """calculate weights"""
                network_status = forward_calculate(data, wHid, wOut)
                wHid, wOut = back_calculate(network_status, wHid, wOut, eta, label)

        """ save weights """
        np.save("./data/wHid.npy", wHid)
        np.save("./data/wOut.npy", wOut)
        
    return wHid, wOut

                                        
def test(test_data, wHid, wOut):
    """ Test perceptron
    Function Inputs:
    test (array)
    """
    data_num = test_data.shape[0]
    test_data = np.hstack((np.ones((data_num, 1)), test_data))
    pre_label = []

    print "\nResult:", "Input", "Score", "Label"
    for t in test_data:
        network_status = forward_calculate(t, wHid, wOut)
        score =  network_status["output_act"]
        label = np.argmax(score)
        print t[1:], score, label


                         
    
def forward_calculate(input, wHid, wOut):
    """ calculate forward process
    Input wHid: weight of hidden layer, wOut: weight of output layer
    Output netwrok_status:dictionary of network status
    """
    
    hidden_base = #TODO#
    hidden_act = #TODO#
    output_base = #TODO#
    output_act = #TODO#
    network_status={"input" : input,
                    "hidden_base" : hidden_base,
                    "hidden_act" : hidden_act,
                    "output_base" : output_base,
                    "output_act" : output_act
                    }

    return network_status

def back_calculate(network_status, wHid, wOut, eta, label):
    """ calculate back-propagation and  update parameters """ 
    input, hidden_base, hidden_act, output_act =\
       network_status["input"], network_status["hidden_base"], network_status["hidden_act"], network_status["output_act"] 
    deltaOut = #TODO#
    deltaHid = #TODO#
    wHid -= #TODO#
    wOut -= #TODO
                           
    return wHid, wOut


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        
def softmax(x):
    return np.exp(x) / np.sum( np.exp(x) )
           
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

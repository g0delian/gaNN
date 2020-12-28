import numpy as np 
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os.path

# target function
def target_function(x1, x2):
    return math.sin(2*x1 + 2.0)*(math.cos(0.5*x2) + 0.5)

def main():
    # sample values (0, 360)
    sample_a = sample_a = np.random.randint(361, size=(21, 2))

    # target values
    target_l = []

    train_f = open("train.dat", "w")
    test_f = open("test.dat", "w")

    # calculate target value
    for row in sample_a:
        x1 = row[0]
        x2 = row[1]
        target_value = target_function(x1, x2)
        target_l.append(target_value)

    # randomly seperate training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        sample_a, target_l, test_size=0.45, random_state=42) # 11 for training 10 for test

    # write to train.dat and test.dat in the requested format
    for i in range(len(X_train)):
        train_f.write("{} {} ".format(str(X_train[i][0]), str(X_train[i][1])))
        train_f.write(str(y_train[i]))
        train_f.write("\n")

    for i in range(len(X_test)):
        test_f.write(str(X_test[i]) + " ")
        test_f.write(str(y_test[i]))
        test_f.write("\n")

    # draw the graphics for the training set

    # create the main figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # add the subplot for training set
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.title('Training set scatter plot')
    zdata = y_train
    x1data = X_train[:,0]
    x2data = X_train[:,1]
    training = ax.scatter3D(x1data, x2data, zdata, c=zdata, cmap='Reds')
    
    # add the subplot for test set
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plt.title('Test set scatter plot')
    zdata = y_test
    x1data = X_test[:,0]
    x2data = X_test[:,1]
    test = ax.scatter3D(x1data, x2data, zdata, c=zdata, cmap='Blues')
    
    # show the main figure
    plt.show()


if __name__ == "__main__":
    main()
    # if os.path.isfile("train.dat") and os.path.isfile("test.dat"):
    #     print("Please remove train.dat and test.dat to generate data.")
    # else:
    #     main()
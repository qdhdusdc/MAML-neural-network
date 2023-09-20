import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data
Y = diabetes.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y.reshape(-1,1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lr = 0.001  #learning rate of maml
lr1 = 0.002  #learning rate of testing

input_dim = len(X[0])
hidden1_dim = 20
hidden2_dim = 20
out_dim = 1

epoches = 100
tasks = 50
beta = 0.01

D_points_train = 120 # Size of training points set for maml
D_points_test = 10 #Size of testing points set for maml

def d_tanh(x):
    return 1 - np.tanh(x)**2

def get_sample(num):#从训练集取采样点
    random_indices = np.random.choice(len(X_train), size=num, replace=False)
    return X_train[random_indices],Y_train[random_indices]



class MamlModel:
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,out_dim):
        super(MamlModel, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.out_dim = out_dim
        self.W1 = np.zeros((input_dim,hidden1_dim))
        self.W2 = np.zeros((hidden1_dim,hidden2_dim))
        self.W3 = np.zeros((hidden2_dim,out_dim))
        self.b1 = np.zeros((1,hidden1_dim))
        self.b2 = np.zeros((1,hidden2_dim))
        self.b3 = np.zeros((1,out_dim))

def forward(X_train,W1,W2,W3,b1,b2,b3):
    h1 = np.dot(X_train,W1) + b1
    h1 = np.tanh(h1)
    h2 = np.dot(h1,W2) + b2
    h2 = np.tanh(h2)
    Y_predict = np.dot(h2,W3) + b3
    Y_predict = np.tanh(Y_predict)
    return Y_predict

def loss_function(y_pred,y_true):
    result=0
    for i in range(0,len(y_pred)):
        result+=(y_pred[i]-y_true[i][0])**2
    return 0.5*result

maml = MamlModel(input_dim,hidden1_dim,hidden2_dim,out_dim)
maml_W1 = np.random.rand(input_dim,hidden1_dim)
maml_W2 = np.random.rand(hidden1_dim,hidden2_dim)
maml_W3 = np.random.rand(hidden2_dim,out_dim)
maml_b1 = np.random.rand(1,hidden1_dim)
maml_b2 = np.random.rand(1,hidden2_dim)
maml_b3 = np.random.rand(1,out_dim)
maml_loss = []

train_maml_W1 = np.random.rand(tasks,input_dim,hidden1_dim)
train_maml_W2 = np.random.rand(tasks,hidden1_dim,hidden2_dim)
train_maml_W3 = np.random.rand(tasks,hidden2_dim,out_dim)
train_maml_b1 = np.random.rand(tasks,1,hidden1_dim)
train_maml_b2 = np.random.rand(tasks,1,hidden2_dim)
train_maml_b3 = np.random.rand(tasks,1,out_dim)

meta_gradient_W1=np.random.rand(input_dim,hidden1_dim)
meta_gradient_W2=np.random.rand(hidden1_dim,hidden2_dim)
meta_gradient_W3=np.random.rand(hidden2_dim,out_dim)
meta_gradient_b1=np.random.rand(1,hidden1_dim)
meta_gradient_b2=np.random.rand(1,hidden2_dim)
meta_gradient_b3=np.random.rand(1,out_dim)

def MultiLayerNN(x_train,Y_train,epoch,lr):
    global maml_W1,maml_W2,maml_W3,maml_b1,maml_b2,maml_b3
    W1,W2,W3,b1,b2,b3=maml_W1.copy(),maml_W2.copy(),maml_W3.copy(),maml_b1.copy(),maml_b2.copy(),maml_b3.copy()
    L=[]
    X_train = x_train.reshape(1,len(x_train))
    for i in range(0,epoch):
        h1 = np.dot(X_train,W1)+b1
        h1 = np.tanh(h1)
        h2 = np.dot(h1,W2)+b2
        h2 = np.tanh(h2)
        Y_predict = np.dot(h2,W3)+b3
        Y_predict = np.tanh(Y_predict)
        loss = loss_function(Y_train,Y_predict)
        L.append(loss)
        dh2 = np.dot((Y_predict-Y_train)*d_tanh(Y_predict),W3.T)
        dh1 = np.dot(dh2*d_tanh(h1),W2.T)
        db1 = dh1*d_tanh(h1)
        db2 = dh2*d_tanh(h2)
        db3 = (Y_predict-Y_train)*d_tanh(Y_predict)
        dW1 = np.dot(X_train.T,dh1*d_tanh(h1))
        dW2 = np.dot(h1.T,dh2*d_tanh(h2))
        dW3 = np.dot(h2.T,(Y_predict-Y_train)*d_tanh(Y_predict))
        W1-=lr*dW1
        W2-=lr*dW2
        W3-=lr*dW3
        b1-=lr*db1
        b2-=lr*db2
        b3-=lr*db3
    return X_train,Y_predict,L

def MultiLayerNN_Random_Parameter(x_train,Y_train,epoch,lr):
    X_train = x_train.reshape(1,len(x_train))
    W1 = np.random.rand(input_dim,hidden1_dim)
    W2 = np.random.rand(hidden1_dim,hidden2_dim)
    W3 = np.random.rand(hidden2_dim,out_dim)
    b1 = np.random.rand(1,hidden1_dim)
    b2 = np.random.rand(1,hidden2_dim)
    b3 = np.random.rand(1,out_dim)
    L = []
    for i in range(0,epoch):
        h1 = np.dot(X_train,W1)+b1
        h1 = np.tanh(h1)
        h2 = np.dot(h1,W2)+b2
        h2 = np.tanh(h2)
        Y_predict = np.dot(h2,W3)+b3
        Y_predict = np.tanh(Y_predict)
        loss = loss_function(Y_train,Y_predict)
        L.append(loss)
        dh2 = np.dot((Y_predict-Y_train)*d_tanh(Y_predict),W3.T)
        dh1 = np.dot(dh2*d_tanh(h1),W2.T)
        db1 = dh1*d_tanh(h1)
        db2 = dh2*d_tanh(h2)
        db3 = (Y_predict-Y_train)*d_tanh(Y_predict)
        dW1 = np.dot(X_train.T,dh1*d_tanh(h1))
        dW2 = np.dot(h1.T,dh2*d_tanh(h2))
        dW3 = np.dot(h2.T,(Y_predict-Y_train)*d_tanh(Y_predict))
        W1-=lr*dW1
        W2-=lr*dW2
        W3-=lr*dW3
        b1-=lr*db1
        b2-=lr*db2
        b3-=lr*db3
    return X_train,Y_predict,L

#maml training
def train(epoch):
    #Training on each task and retain the parameters
    global meta_gradient_W1,meta_gradient_W2,meta_gradient_W3,meta_gradient_b1,meta_gradient_b2,meta_gradient_b3
    global maml_W1,maml_W2,maml_W3,maml_b1,maml_b2,maml_b3,train_maml_W1,train_maml_W2,train_maml_W3,train_maml_b1,train_maml_b2,train_maml_b3
    global maml_loss
    loss_sum = 0.0
    for i in range(tasks):
        maml.W1,maml.W2,maml.W3,maml.b1,maml.b2,maml.b3 = maml_W1,maml_W2,maml_W3,maml_b1,maml_b2,maml_b3
        maml_x_train,maml_y_train = get_sample(D_points_train)
        maml_x_test,maml_y_test = get_sample(D_points_test)
        for d in range(len(maml_x_train)):
            x_train = maml_x_train[d].reshape(-1,len(maml_x_train[0]))
            Y_predict = forward(x_train,maml.W1,maml.W2,maml.W3,maml.b1,maml.b2,maml.b3)
            h1 = np.dot(x_train,maml.W1)+maml.b1
            h1 = np.tanh(h1)
            h2 = np.dot(h1,maml.W2)+maml.b2
            h2 = np.tanh(h2)
            dh2 = np.dot((Y_predict-maml_y_train[d])*d_tanh(Y_predict),maml.W3.T)
            dh1 = np.dot(dh2*d_tanh(h1),maml.W2.T)
            db1 = dh1*d_tanh(h1)
            db2 = dh2*d_tanh(h2)
            db3 = (Y_predict-maml_y_train[d])*d_tanh(Y_predict)
            dW1 = np.dot(x_train.T,dh1*d_tanh(h1))
            dW2 = np.dot(h1.T,dh2*d_tanh(h2))
            dW3 = np.dot(h2.T,(Y_predict-maml_y_train[d])*d_tanh(Y_predict))

            maml.W1-=lr*dW1
            maml.W2-=lr*dW2
            maml.W3-=lr*dW3
            maml.b1-=lr*db1
            maml.b2-=lr*db2
            maml.b3-=lr*db3
            train_maml_W1[i] = maml.W1
            train_maml_W2[i] = maml.W2
            train_maml_W3[i] = maml.W3
            train_maml_b1[i] = maml.b1
            train_maml_b2[i] = maml.b2
            train_maml_b3[i] = maml.b3

        for d in range(len(maml_x_test)):
            x_test = maml_x_test[d].reshape(-1,len(maml_x_test[0]))
            Y_predict = forward(x_test,maml.W1,maml.W2,maml.W3,maml.b1,maml.b2,maml.b3)
            maml.W1 = train_maml_W1[i]
            maml.W2 = train_maml_W2[i]
            maml.W3 = train_maml_W3[i]
            maml.b1 = train_maml_b1[i]
            maml.b2 = train_maml_b2[i]
            maml.b3 = train_maml_b3[i]
            
            loss_value = loss_function(maml_y_test[d], Y_predict)
            loss_sum = loss_sum + loss_value
            
            meta_gradient_W1 = meta_gradient_W1 + lr*dW1
            meta_gradient_W2 = meta_gradient_W2 + lr*dW2
            meta_gradient_W3 = meta_gradient_W3 + lr*dW3
            meta_gradient_b1 = meta_gradient_b1 + lr*db1
            meta_gradient_b2 = meta_gradient_b2 + lr*db2
            meta_gradient_b3 = meta_gradient_b3 + lr*db3

    maml.W1 -= beta * meta_gradient_W1 / tasks
    maml.W2 -= beta * meta_gradient_W2 / tasks
    maml.W3 -= beta * meta_gradient_W3 / tasks
    maml.b1 -= beta * meta_gradient_b1 / tasks
    maml.b2 -= beta * meta_gradient_b2 / tasks
    maml.b3 -= beta * meta_gradient_b3 / tasks
    maml_loss.append(loss_sum/tasks)
    if  epoch%10==0:
        print("the Epoch is {:04d}".format(epoch),"the Loss is {:.4f}".format(loss_sum/tasks))

if __name__ == "__main__":
    for epoch in range(epoches):
        train(epoch)

    plt.plot(range(len(maml_loss)),maml_loss,label='MAML Loss')
    plt.legend()
    plt.savefig('MAML_loss.png',dpi=300)
    plt.close()
    
    sample_x1,sample_y1,sample_loss1 = [],[],[]
    sample_x2,sample_y2,sample_loss2 = [],[],[]
    for i in range(0,len(X_test)):
        x1,y1,L1 = MultiLayerNN(X_test[i],Y_test[i],100,lr1)
        x2,y2,L2 = MultiLayerNN_Random_Parameter(X_test[i],Y_test[i],100,lr1)
        sample_x1.append(x1)
        sample_x2.append(x2)
        sample_y1.append(scaler.inverse_transform(y1[0]))
        sample_y2.append(scaler.inverse_transform(y2[0]))
        sample_loss1.append(L1)
        sample_loss2.append(L2)
    loss1 = np.mean(sample_loss1,axis=0)
    loss2 = np.mean(sample_loss2,axis=0)
    plt.plot(range(len(loss1)),loss1,label='Loss(with MAML)')
    plt.plot(range(len(loss2)),loss2,label='Loss(without MAML)')
    plt.legend()
    plt.savefig('loss.png',dpi=300)
    plt.close()
    
    sample_y1 = np.array(sample_y1).reshape(1,-1)[0]
    sample_y2 = np.array(sample_y2).reshape(1,-1)[0]
    Y_test = scaler.inverse_transform(Y_test).reshape(1,-1)[0]
    sorted_indices = np.argsort(Y_test)
    plt.plot(range(len(sample_y1)),sample_y1[sorted_indices],label='Predicted result(with MAML)')
    plt.plot(range(len(sample_y2)),sample_y2[sorted_indices],label='Predicted result(without MAML)')
    plt.plot(range(len(Y_test)),Y_test[sorted_indices],label='True value')
    plt.legend()
    plt.savefig('Predicted result.png',dpi=300)
    plt.close()
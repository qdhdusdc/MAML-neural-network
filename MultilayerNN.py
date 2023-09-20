import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.transforms import Bbox

lba = 0.1
uba = 10
lbb = -np.pi
ubb = np.pi
a1 = np.random.uniform(lba,uba)
b1 = np.random.uniform(lbb,ubb)
lb,ub = -5,5
lr = 0.0001
lr1 = 0.002
input_dim = 1
hidden1_dim = 20
D_dim = hidden1_dim
out_dim = 1
epochs = 500
tasks = 50
beta = 0.01
D_points_train = 10
D_points_test = 5
points = 10

def samplePoints(k,test=None):#test=true for maml training, a,b will update in every iteration. Else a,b are global variables for test.
    x = (ub-lb)*np.random.rand(k,input_dim)+lb
    if test==True:
        y = a1 * np.sin(x + b1)
        return x,y
    a = np.random.uniform(lba,uba)
    b = np.random.uniform(lbb,ubb)
    y = a * np.sin(x + b)
    return x,y

def maml_samplePoints(a,b):
    x = (ub-lb)*np.random.rand(1,input_dim)+lb
    y = a * np.sin(x + b)
    return x,y

class MamlModel:
    def __init__(self,input_dim,hidden1_dim,out_dim):
        super(MamlModel, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.out_dim = out_dim
        self.W1 = np.zeros((input_dim,hidden1_dim))
        self.W2 = np.ones((hidden1_dim,out_dim))
        self.D = np.diag(np.random.rand(D_dim))
        self.b1 = np.zeros((1,hidden1_dim))
        self.b2 = np.zeros((1,out_dim))

def forward(X_train,W1,b1,D,b2):
    W2 = np.ones((hidden1_dim,out_dim))
    hidden1 = np.dot(X_train,W1) + b1
    Y_predict = np.dot(np.dot(hidden1,D),W2) + b2
    return Y_predict

def loss_function(y_pred,y_true):
    result=0
    for i in range(0,len(y_pred)):
        result+=(y_pred[i][0]-y_true[i][0])**2
    return 0.5*result

maml = MamlModel(input_dim,hidden1_dim,out_dim)
maml_W1 = np.random.rand(input_dim,hidden1_dim)
maml_D = np.diag(np.random.rand(D_dim))
maml_W2 = np.ones((hidden1_dim,out_dim))
maml_b1 = np.random.rand(1,hidden1_dim)
maml_b2 = np.random.rand(1,out_dim)

train_maml_W1 = np.random.rand(tasks,input_dim,hidden1_dim)
train_maml_D = []
for i in range(tasks):
    train_maml_D.append(np.diag(np.random.rand(D_dim)))
train_maml_b1 = np.random.rand(tasks,1,hidden1_dim)
train_maml_b2 = np.random.rand(tasks,1,out_dim)

meta_gradient_W1=np.random.rand(input_dim,hidden1_dim)
meta_gradient_D=np.diag(np.random.rand(D_dim))
meta_gradient_b1=np.random.rand(1,hidden1_dim)
meta_gradient_b2=np.random.rand(1,out_dim)

def SimpleLayerNN(X_train,Y_train,epoch,lr):
    global maml_W1,maml_b1,maml_b2
    W1,W2,D,b1,b2=maml_W1.copy(),maml_W2.copy(),maml_D.copy(),maml_b1.copy(),maml_b2.copy()
    h1 = np.dot(X_train,W1)+b1
    h2 = np.dot(h1,D)
    Y_predict = np.dot(h2,W2)+b2
    
    for i in range(1,epoch):
        h1 = np.dot(X_train,W1)+b1
        h2 = np.dot(h1,D)
        Y_predict = np.dot(h2,W2)+b2
        loss = loss_function(Y_train,Y_predict)
        if i%100 == 0:
            print("Epoch:{:d}".format(i), "Loss:%s"%loss)
        h1 = np.dot(X_train,maml.W1)+maml.b1
        h2 = np.dot(h1,maml.D)
        dh2 = np.dot((Y_predict-Y_train),maml.W2.T)
        dh1 = np.dot(dh2,maml.D.T)
        db2 = Y_predict-Y_train
        db1 = dh1
        dD = np.dot(h1.T,dh2)
        dW1 = np.dot(X_train.T,dh1)
        W1-=lr*dW1
        b1-=lr*db1
        b2-=lr*db2
        D-=lr*dD
    return X_train,Y_predict

def SimpleLayerNN_Random_Parameter(X_train,Y_train,epoch,lr):
    W1 = np.random.rand(input_dim,hidden1_dim)
    W2 = np.ones((hidden1_dim,out_dim))
    D = np.diag(np.random.rand(D_dim))
    b1 = np.random.rand(1,hidden1_dim)
    b2 = np.random.rand(1,out_dim)
    h1 = np.dot(X_train,W1)+b1
    h2 = np.dot(h1,D)
    Y_predict = np.dot(h2,W2)+b2
     
    for i in range(1,epoch):
        h = np.dot(X_train,W1)+b1
        Y_predict = np.dot(h,W2)+b2
        loss = loss_function(Y_train,Y_predict)
        if i%100 == 0:
            print("Epoch:{:d}".format(i), "Loss:%s"%loss)
        h1 = np.dot(X_train,maml.W1)+maml.b1
        h2 = np.dot(h1,maml.D)
        dh2 = np.dot((Y_predict-Y_train),maml.W2.T)
        dh1 = np.dot(dh2,maml.D.T)
        db2 = Y_predict-Y_train
        db1 = dh1
        dD = np.dot(h1.T,dh2)
        dW1 = np.dot(X_train.T,dh1)
        W1-=lr*dW1
        b1-=lr*db1
        b2-=lr*db2
        D-=lr*dD
    return X_train,Y_predict

#maml training
def train(epoch):
    #Training on each task and retain the parameters
    global meta_gradient_W1,meta_gradient_D,meta_gradient_b1,meta_gradient_b2
    global maml_W1,maml_b1,maml_b2,maml_D,train_maml_D,train_maml_W1,train_maml_W2,train_maml_b1,train_maml_b2
    loss_sum = 0.0
    for i in range(tasks):
        maml.W1,maml.b1,maml.b2,maml.D = maml_W1,maml_b1,maml_b2,maml_D
        a = np.random.uniform(lba,uba)
        b = np.random.uniform(lbb,ubb)
        for d in range(D_points_train):
            X_train,Y_train = maml_samplePoints(a,b)#生成maml的训练点
            
            Y_predict = forward(X_train,maml.W1,maml.b1,maml.D,maml.b2)#神经网络预测结果
            h1 = np.dot(X_train,maml.W1)+maml.b1
            h2 = np.dot(h1,maml.D)
            dh2 = np.dot((Y_predict-Y_train),maml.W2.T)
            dh1 = np.dot(dh2,maml.D.T)
            db2 = Y_predict-Y_train
            db1 = dh1
            dD = np.dot(h1.T,dh2)
            dW2 = np.dot(h2.T,(Y_predict-Y_train))
            dW1 = np.dot(X_train.T,dh1)
            
            maml.W1-=lr*dW1
            maml.b1-=lr*db1
            maml.b2-=lr*db2
            maml.D-=lr*dD
            train_maml_W1[i] = maml.W1
            train_maml_b1[i] = maml.b1
            train_maml_b2[i] = maml.b2
            train_maml_D[i] = maml.D

        for d in range(D_points_test):
            X_test,Y_test = maml_samplePoints(a,b)
            Y_predict = forward(X_test,maml.W1,maml.b1,maml.D,maml.b2)
            maml.W1 = train_maml_W1[i]
            maml.b1 = train_maml_b1[i]
            maml.b2 = train_maml_b2[i]
            maml.D = train_maml_D[i]
            
            h1 = np.dot(X_test,maml.W1)+maml.b1
            h2 = np.dot(h1,maml.D)
            dh2 = np.dot((Y_predict-Y_test),maml.W2.T)
            dh1 = np.dot(dh2,maml.D.T)
            db2 = Y_predict-Y_test
            db1 = dh1
            dD = np.dot(h1.T,dh2)
            dW1 = np.dot(X_test.T,dh1)
            
            maml.W1-=lr*dW1
            maml.W2-=lr*dW2
            maml.b1-=lr*db1
            maml.b2-=lr*db2
            maml.D-=lr*dD

            loss_value = loss_function(Y_test, Y_predict)
            loss_sum = loss_sum + loss_value
            
            meta_gradient_W1 = meta_gradient_W1 + lr*dW1
            meta_gradient_b1 = meta_gradient_b1 + lr*db1
            meta_gradient_b2 = meta_gradient_b2 + lr*db2
            meta_gradient_D = meta_gradient_D + lr*dD
            
    maml.W1 -= beta * meta_gradient_W1 / tasks
    maml.b1 -= beta * meta_gradient_b1 / tasks
    maml.b2 -= beta * meta_gradient_b2 / tasks
    maml.D -= beta*meta_gradient_D / tasks
    if  epoch%100==0:
        print("the Epoch is {:04d}".format(epoch),"the Loss is {:.4f}".format(loss_sum/tasks))

def fit_func(x,a,b):
    return a*np.sin(x+b)

if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
    X_true,Y_true,X1,Y1,X2,Y2,X3,Y3,X4,Y4,X5,Y5,X6,Y6=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for i in range(0,points):
        x,y = samplePoints(1,test=True)
        x1,y1 = SimpleLayerNN(x,y,0,lr1)
        x2,y2 = SimpleLayerNN(x,y,3,lr1)
        x3,y3 = SimpleLayerNN(x,y,10,lr1)
        x4,y4 = SimpleLayerNN_Random_Parameter(x,y,0,lr1)
        x5,y5 = SimpleLayerNN_Random_Parameter(x,y,3,lr1)
        x6,y6 = SimpleLayerNN_Random_Parameter(x,y,10,lr1)        
        X_true.append(x)
        Y_true.append(y)
        X1.append(x1[0][0])
        Y1.append(y1[0][0])
        X2.append(x2[0][0])
        Y2.append(y2[0][0])
        X3.append(x3[0][0])
        Y3.append(y3[0][0])
        X4.append(x4[0][0])
        Y4.append(y4[0][0])
        X5.append(x5[0][0])
        Y5.append(y5[0][0])
        X6.append(x6[0][0])
        Y6.append(y6[0][0])        
        
    plt.scatter(X_true, Y_true, marker='^',s=25,alpha=0.7)
    x_true = np.linspace(lb, ub, 200)
    y_true = [a1*np.sin(xi+b1) for xi in x_true]

    params1, params_covariance1 = curve_fit(fit_func, X1, Y1)
    params2, params_covariance2 = curve_fit(fit_func, X2, Y2)
    params3, params_covariance3 = curve_fit(fit_func, X3, Y3)
    xy_sorted = sorted(zip(X1, Y1))
    x1 = [e[0] for e in xy_sorted]
    y1 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(X2, Y2))
    x2 = [e[0] for e in xy_sorted]
    y2 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(X3, Y3))
    x3 = [e[0] for e in xy_sorted]
    y3 = [e[1] for e in xy_sorted]      
    
    params4, params_covariance4 = curve_fit(fit_func, X4, Y4)
    params5, params_covariance5 = curve_fit(fit_func, X5, Y5)
    params6, params_covariance6 = curve_fit(fit_func, X6, Y6)
    xy_sorted = sorted(zip(X4, Y4))
    x4 = [e[0] for e in xy_sorted]
    y4 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(X5, Y5))
    x5 = [e[0] for e in xy_sorted]
    y5 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(X6, Y6))
    x6 = [e[0] for e in xy_sorted]
    y6 = [e[1] for e in xy_sorted]      
    
    y1 = [params1[0]*np.sin(xi+params1[1]) for xi in x_true]
    y2 = [params2[0]*np.sin(xi+params2[1]) for xi in x_true]
    y3 = [params3[0]*np.sin(xi+params3[1]) for xi in x_true]
    y4 = [params4[0]*np.sin(xi+params4[1]) for xi in x_true]
    y5 = [params5[0]*np.sin(xi+params5[1]) for xi in x_true]
    y6 = [params6[0]*np.sin(xi+params6[1]) for xi in x_true]
    
    plt.plot(x_true,y_true,color='red', label='True Function')
    plt.plot(x_true,y1,color='yellow', label='After 0 Steps(With maml)')
    plt.plot(x_true,y2,color='green', label='After 3 Steps(With maml)')
    plt.plot(x_true,y3,color='blue',label='After 10 Steps(With maml)')
    plt.plot(x_true,y4,color='yellow', linestyle='--',label='After 0 Steps(Without maml)')
    plt.plot(x_true,y5,color='green', linestyle='--',label='After 3 Steps(Without maml)')
    plt.plot(x_true,y6,color='blue', linestyle='--',label='After 10 Steps(Without maml)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
    plt.savefig("MultilayerNN.png",dpi=300,bbox_inches=Bbox.from_bounds(*(0,0,8.5,4)))
    plt.show()
    plt.close('all')
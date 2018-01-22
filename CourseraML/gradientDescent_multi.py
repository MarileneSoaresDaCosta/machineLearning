#gradient descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# function to pause the script to see each plot
def pause():
    programPause = input("Press the <ENTER> key to continue...")

df = pd.read_csv('ex1data2.txt', names = ["size", "beds", "price"])
print('Data uploaded! Table shape:', df.shape)
print('Top data points:')
print(df.head())



# extracting values into array forms
xOne_list = df["size"].tolist()
mu_xOne = df["size"].mean()
std_xOne = df["size"].std()


xTwo_list = df["beds"].tolist()
mu_xTwo = df["beds"].mean()
std_xTwo = df["beds"].std()


y_list = df["price"].tolist()
mu_y = df["price"].mean()
std_y = df["price"].std()

# means and stdev matrices for feature normalization
mu = [1, mu_xOne, mu_xTwo]

sigma = [0, std_xOne, std_xTwo]

print('mu matrix', mu)
print('sigma matrix', sigma)


# plotting data
# size and price
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('House size and price')

ax.set_xlabel('House size (sqfeet)')
ax.set_ylabel('House price')
plt.plot(df["size"], df["price"], 'rx')
plt.show(block=False)
plt.savefig("Multi_sizeAndPriceData.png")

pause()

# num of bedrooms and price
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Number of bedrooms and price')

ax.set_xlabel('Number of bedrooms')
ax.set_ylabel('House price')
plt.plot(df["beds"], df["price"], 'rx')
plt.show(block=False)
plt.savefig("Multi_bedsAndPriceData.png")
pause()



# setting up the design matrix

m = len(y_list)
n = len(df.columns) - 1 # 2 features, size and beds - price is y
# print('number of features', n)
iterations = 1500
alpha = 0.01


X = np.vstack([np.ones(m), xOne_list, xTwo_list]).T
y = np.vstack([y_list]).T

# initializing theta with zeros
theta = np.zeros((n+1,1)) 
print('X.shape ', X.shape, '\n')
print('theta.shape ', theta.shape, '\n')




''' FEATURENORMALIZE Normalizes the features in X 
FEATURENORMALIZE(X) returns a normalized version of X where
the mean value of each feature is 0 and the standard deviation
is 1. This is often a good preprocessing step to do when
working with learning algorithms. '''

# function  featureNormalize(X) returns [X_norm, mu, sigma] 

def featureNormalize(X, mu, sigma):
    X_norm = X
    # print('X_norm shape: ', X_norm.shape, '\n')

    rows = X.shape[0]
    cols = X.shape[1]

    print('rows, cols',rows, cols)

    for r in range(rows):
        for c in range(cols):
            # print('r: ', r, 'c: ', c, 'Xrc: ', X[r][c])
            if c == 0:
                X_norm[r][c] = 1
            else:
                X_norm[r][c] = (X[r][c] - mu[c]) / sigma[c]
    return(X_norm)

X_norm = featureNormalize(X, mu, sigma)

# plotting NORMALIZED data
# size and price
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('House size (NORMALIZED) and price', fontweight='bold', color='green')

ax.set_xlabel('House size (sqfeet)')
ax.set_ylabel('House price')
plt.plot(X_norm[:,1], y, 'gx')
plt.show(block=False)
plt.savefig("Multi_sizeAndPriceData_norm.png")

pause()

# num of bedrooms and price
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Number of bedrooms (NORMALIZED) and price', fontweight='bold', color='green')

ax.set_xlabel('Number of bedrooms')
ax.set_ylabel('House price')
plt.plot(X_norm[:,2], y, 'gx')
plt.show(block=False)
plt.savefig("Multi_bedsAndPriceData_norm.png")
pause()

# GRADIENT DESCENT 
print("Calculating Gradient Descent... \n")
alpha = float(input('enter alpha: '))
iterations = int(input('enter number of iterations: '))
print('you chose alpha= ', alpha, " and iterations = ", iterations)


def computeCost(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    # print('predictions shape', predictions.shape)

    Errors = predictions - y
    # print('Errors.shape',Errors.shape)

    sqrErrors = np.power(Errors, 2)
    # print('sqrErrors.shape',sqrErrors.shape)

    J = 1 / (2 * m) * np.sum(sqrErrors)
    return(J)


print('checking matrices dimensions before running Gradient Descent\n')
print('X shape ', X.shape, '\n')
print('y shape', y.shape, '\n')
print('theta shape', theta.shape, '\n')
print('current theta \n', theta, '\n')

print('Running Gradient Descent with alpha ', alpha, ' and ', iterations, ' iterations \n')

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    
    for i in range(0,num_iters):
        theta = theta - (np.dot(np.dot(np.dot((np.dot(X, theta) - y).T, X), (1/m)), alpha)).T
        J_history[i] = computeCost(X, y, theta)
        i += 1
    
    return theta, J_history


theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print('J_history outside', J_history)

print('Theta found by Gradient Descent: \n',theta)


# find the predicted price using gradient descent
# but, first we normalize size 1650 and 3 bedrooms
x1 = (1650 - mu_xOne)/ std_xOne
x2 = (3 - mu_xTwo) / std_xTwo
priceGD = theta[0][0] + theta[1][0] * x1 + theta[2][0] * x2


# plotting the J_history convergence graph
# print('j shape', J_history.shape)
# print(J_history)
print('Plotting the covergence of the Gradient Descent for alpha ', alpha, ' and ', iterations, ' interations')


# caculate new results for various learning rates
iterations = 50
alpha = 0.01
theta = np.zeros((n+1,1)) 
theta, J_history_0p01 = gradientDescent(X, y, theta, alpha, iterations)


alpha = 0.03
theta = np.zeros((n+1,1)) 
theta, J_history_0p03 = gradientDescent(X, y, theta, alpha, iterations)



alpha = 0.1
theta = np.zeros((n+1,1)) 
theta, J_history_0p1 = gradientDescent(X, y, theta, alpha, iterations)

alpha = 0.3
theta = np.zeros((n+1,1)) 
theta, J_history_0p3 = gradientDescent(X, y, theta, alpha, iterations)

alpha = 1
theta = np.zeros((n+1,1)) 
theta, J_history_1 = gradientDescent(X, y, theta, alpha, iterations)

alpha = 1.25
theta = np.zeros((n+1,1)) 
theta, J_history_1p25 = gradientDescent(X, y, theta, alpha, iterations)


alpha = 1.35
theta = np.zeros((n+1,1)) 
theta, J_history_1p35 = gradientDescent(X, y, theta, alpha, iterations)


# plotting the J(theta) curves


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
fig3.subplots_adjust(top=0.85)
ax3.set_title('Convergence of Gradient Descent')

ax3.set_xlabel('Number of Iterations')
ax3.set_ylabel('Cost J')

a = np.arange(0, iterations)
# print('a shape', a.shape)
axes = plt.gca()
axes.set_ylim([0,0.8e11])

J0p01, = plt.plot(a, J_history_0p01[:,0], 'b-', label="a = 0.01")

J0p03, = plt.plot(a, J_history_0p03[:,0], 'g-', label="a = 0.03")

J0p1,  = plt.plot(a, J_history_0p1[:,0], 'r-', label="a = 0.1")

J0p3,  = plt.plot(a, J_history_0p3[:,0], 'c-', label="a = 0.3")

J1, = plt.plot(a, J_history_1[:,0], 'm-', label="a = 1")

J1p25, = plt.plot(a, J_history_1p25[:,0], 'y-', label="a = 1.25")

J1p35, = plt.plot(a, J_history_1p35[:,0], 'k-', label="a = 1.35")


plt.legend(handles=[J0p01, J0p03, J0p1, J0p3, J1, J1p25, J1p35])


plt.show(block=False)
plt.savefig("J_history_multi_alphaTest.png")

pause()


# Normal Equation
print('Solving with normal equations...\n')

theta = np.zeros((n+1,1)) 

X = np.vstack([np.ones(m), xOne_list, xTwo_list]).T


def normalEq(X, y):
    theta = np.dot( np.dot( np.linalg.pinv( np.dot( X.T, X )), X.T), y)
    return(theta)
# print (X)
print('theta calculated by normal equation: ',normalEq(X, y), '\n')

print('predicting the price of a house of size 1,650sqfeet and 3 bedrooms \n')
theta = normalEq(X, y)
x1 = 1650
x2 = 3
print(theta[0][0])
price = theta[0][0] + theta[1][0] * x1 + theta[2][0] * x2

print('price found by normal Equation = ',price, '\n')
print('price found by gradient descent = ',priceGD, '\n')





print('end of script')
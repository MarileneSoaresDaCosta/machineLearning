#gradient descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# function to pause the script to see each plot
def pause():
    programPause = input("Press the <ENTER> key to continue...")

df = pd.read_csv('ex1data1.txt', names = ["pop", "profit"])
print('Data uploaded! Table shape:', df.shape)
print('Top data points:')
print(df.head())



# extracting values into array forms
x_list = df["pop"].tolist()
y_list = df["profit"].tolist()


# plotting data
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Scatter plot of training data')

ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000s')
plt.plot(df["pop"], df["profit"], 'rx')
plt.show(block=False)
plt.savefig("data_Scatter.png")

pause()

# settings
m = len(x_list)
n = 1
iterations = 1500
alpha = 0.01

# design matrix
X = np.vstack([np.ones(m), x_list]).T
y = np.vstack([y_list]).T

# initializing theta with zeros
theta = np.zeros((n+1,1)) 

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



# print('checking matrices dimensions \n')
# print('X shape ', X.shape, '\n')
# print('y shape', y.shape, '\n')
# print('theta shape', theta.shape, '\n')

# testing of the cost function
J = computeCost(X, y, theta)
print('Computing cost for theta [0,0]: ', J)
print('Expected cost value (approx) 32.07\n')



# further testing of the cost function
theta[0][0] = -1
theta[1][0] = 2
J = computeCost(X, y, theta);
print('\nWith theta = [-1 ; 2]\nCost computed = ', J);
print('Expected cost value (approx) 54.24\n');


pause()

theta = np.zeros((n+1,1)) 
print('checking matrices dimensions before running Gradient Descent\n')
print('X shape ', X.shape, '\n')
print('y shape', y.shape, '\n')
print('theta shape', theta.shape, '\n')
print('current theta', theta, '\n')

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

print('J_history outside', J_history)

print('Theta found by Gradient Descent: \n',theta)

print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')


# adding regression line to scatter
# print('expected x1, y1 = 5, 2.2017 \n expected x2, y2 = 22, 22.0305')
x1 = 5
y1 = theta[0][0] + theta[1][0] * x1
# print('x1, y1', x1, y1)
x2 = 22
y2 = theta[0][0] + theta[1][0] * x2
# print('x2, y2', x2, y2)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.subplots_adjust(top=0.85)
ax2.set_title('Data and Linear Regression')

ax2.set_xlabel('Population of City in 10,000s')
ax2.set_ylabel('Profit in $10,000')


data, = plt.plot(x_list, y_list, 'rx', label='Training data')
regr_line, = plt.plot([x1, x2], [y1, y2], 'k-', lw=2, label='Linear Regression')
plt.legend(handles=[data, regr_line])
plt.show(block=False)
plt.savefig("data_RegLine.png")

pause()

# plotting the J_history convergence graph
# print('j shape', J_history.shape)
# print(J_history)
print('plotting the covergence of the Gradient Descent')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
fig3.subplots_adjust(top=0.85)
ax3.set_title('Convergence of Gradient Descent')

ax3.set_xlabel('Number of Iterations')
ax3.set_ylabel('Cost J')

a = np.arange(0, iterations)
# print('a shape', a.shape)

plt.plot(a, J_history[:,0])
plt.show(block=False)
plt.savefig("J_history.png")
pause()

print('end of script')
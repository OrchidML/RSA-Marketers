import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

#load data
DATA = '/home/ronald/data/cripps20112014.csv'
df = pd.read_csv(DATA)

#define 'function to predict', in this case, the values we want to 'train' it with
def f(x):
    return np.array(df['value'][x].values)

#define start and end points, and a list of all values between them. p_start and p_end also control graphing limits
#poi are the points of interest to be evaluated from the function to be predicted
p_start, p_end = 105, 156
p = range(p_start, p_end)
poi = np.array([105, 108, 113, 115, 120, 125, 132, 140, 144, 150, 155])

#defining points of interesting as a unidimensional series to index our dataframe in f(x)
X = pd.Series(poi).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d((np.linspace(p_start, p_end, ((p_end-p_start)*4+1)))).T

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, \
                     random_start=5)

# Fit to data using Maximum Likelihood Estimation of the parameters
#R is the same values as X, but in a 2 dimensional np array as necessary for gp.fit()
R = np.atleast_2d(poi).T
gp.fit(R, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

#creating index values k and k1 to line up predicted/actual/error values
#create a resulting dataframe for the values
k = pd.Series(p)
k1 = pd.Series(range(0,p_end-p_start))
result = pd.DataFrame(index=k, columns=['actual', 'GP', 'error'])

# ---------------------------------------------------------------------------
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()

#this plots 'red dots', the observed points ie. our real values taken into consideration
plt.plot(R, y, 'r.', markersize=10, label=u'Observations')

#this plots our 'red line', actual values
plt.plot(range(p_start, p_end), df['value'][p_start:p_end], 'r-', label=u'Actual')

#this plots our 'blue line', the predicted values
plt.plot(x, y_pred, 'b-', label=u'Prediction')

#this plots the light blue area, confidence interval
plt.fill(np.concatenate([x, x[::-1]]), \
       np.concatenate([y_pred - 1.9600 * sigma,
                      (y_pred + 1.9600 * sigma)[::-1]]), \
       alpha=.5, fc='b', ec='None', label='95% confidence interval')

#re-doing the prediction with an input space equal to the predicted range, purely to line up evaluated points
#to the actual points. not proper solution, but work-around to get accurate error reading
x = np.atleast_2d((np.linspace(p_start, p_end, ((p_end-p_start)*1+1)))).T
y_pred, MSE = gp.predict(x, eval_MSE=True)
result['actual'][k] = df['value'][k]
result['GP'][k] = y_pred[k1]
result['error'][k] = abs(y_pred[k1] - df['value'][k])/df['value'][k]

#plot the calculated error, now lining up. multiplied by 10 for easier visualization
plt.plot(k, result['error']*10, 'c-')

#labels and limits
plt.xlabel('$week$')
plt.xlim(p_start,p_end)
plt.ylabel('$value$')
plt.ylim(0,np.max(df['value'][k].values)+1)
plt.legend(loc='lower right')

plt.show()
plt.clf()

#----------------------------------------------------------------------
# evaluating with noise

#noise definition
dy = 0.5 + 0.5 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
print noise
y += noise

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d((np.linspace(p_start, p_end, ((p_end-p_start)*4+1)))).T

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                     thetaL=1e-3, thetaU=1,
                     nugget=(dy / y) ** 0.5,
                     random_start=800)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(R, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
#fig = plt.figure()

plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.plot(k, df['value'][k], 'r-', label=u'Actual')
plt.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')

#re-doing the prediction with an input space equal to the predicted range, purely to line up evaluated points
#to the actual points. not proper solution, but work-around to get accurate error reading
x = np.atleast_2d((np.linspace(p_start, p_end, ((p_end-p_start)*1+1)))).T
y_pred, MSE = gp.predict(x, eval_MSE=True)
result['actual'][k] = df['value'][k]
result['GP'][k] = y_pred[k1]
result['error'][k] = abs(y_pred[k1] - df['value'][k])/df['value'][k]

#plot the calculated error, now lining up. multiplied by 10 for easier visualization
#plt.plot(k, result['error']*10, 'c-')

#labels and limits
plt.xlabel('$week$')
plt.xlim(p_start,p_end)
plt.ylabel('$value$')
plt.ylim(0,np.max(df['value'][k].values)+1)
plt.legend(loc='lower right')

plt.show()
plt.clf()

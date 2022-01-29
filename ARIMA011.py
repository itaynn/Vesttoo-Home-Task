import torch as tch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Computes PDF of normal(0,sigma)
def normalpdf(x, sigma):
    return tch.mul(tch.exp(tch.mul(tch.pow(x, 2), -1 / (2 * sigma * sigma))), 1. / (np.sqrt(2.0 * np.pi) * sigma))

# Computes a series by Y[t]-Y[t-1]=drift+e[t]-theta * e[t-1]
def generatearima011(mu0, theta, drift, N, sigma=1):
    Y = [mu0]
    et_1 = 0
    for t in range(1, N):
        et = np.random.normal(0, sigma)
        Y.append(drift + Y[-1] + et - theta * et_1)
        et_1 = et
    return tch.tensor(Y)


class ARIMA011(tch.nn.Module):
    def __init__(self):
        super(ARIMA011, self).__init__()
        self.theta = Variable(tch.tensor(.5), requires_grad=True)
        self.drift = Variable(tch.tensor(0.5), requires_grad=True)
        self.sigma = Variable(tch.tensor(0.2), requires_grad=True, )
# Find parameters which fit best to the data series, assuming they conform to ARIMA(0,1,1)
    def train(self, data: tch.Tensor):
        opt = tch.optim.LBFGS([self.theta, self.drift, self.sigma], lr=0.1)
        data_stat = tch.diff(data)
        N = len(data_stat)
        err = []
        for itr in range(100):
            # loss = -self.likelihoodarima(sigma, N, theta, drift, data_stat)
            opt.zero_grad()
            objective = -self.likelihoodarima(self.sigma, N, self.theta, self.drift, data_stat)
            objective.backward()
            opt.step(lambda: -self.likelihoodarima(self.sigma, N, self.theta, self.drift, data_stat))
            # opt.step()
        return self.theta, self.drift, self.sigma
    # computes the log likelihood function for the probability of observing the data
    def likelihoodarima(self, sigma, N, theta, drift, data_stat):
        # the formula for the errors:
        # error[0]=0
        # error[i]= data_stat[i] - drift + theta * error[i - 1]
        # error_list = [tch.zeros(1), tch.tensor(data_stat[1] - drift)]
        thetas = theta.repeat(N - 1)
        thetapow = tch.cat((tch.ones(1), tch.cumprod(thetas, dim=0), tch.zeros(N + 1)))
        thetapow = thetapow.roll(1)
        err = -drift * tch.cumsum(thetapow[:N], dim=0)
        for i in range(1, N):
            err = err.add((data_stat[i] * thetapow)[:N])
            thetapow = thetapow.roll(1)
        # the formula for the loglikelihood:
        # -(N-1)*log(sigma)-1/sigma^2*(sum of squares of error[i])
        return tch.add(-N * tch.log(np.sqrt(2 * np.pi) * sigma),
                       tch.sum(tch.mul(tch.pow(err[N - 1], 2), -1.0 / (2 * sigma * sigma))))

    # Computes PDF of the error in the prediction
    def probabilityfuture(self, observations):
        error = [0]
        observations = observations.diff()
        for t in range(1, len(observations)):
            error.append(observations[t] - self.drift + self.theta * error[t - 1])
        return tch.prod(normalpdf(tch.tensor(error), self.sigma)), error


def main():
    N = 20
    mu = 0.0
    theta = 0.2
    drift = 1.0
    sigma = 0.3
    TS = generatearima011(mu, theta, drift, N, sigma)
    print("Estimated parameters: theta=", theta, " drift=", drift, " sigma=", sigma)
    plt.plot(tch.tensor(range(N)), TS)
    plt.show()
    N0 = 14
    model = ARIMA011()
    theta_est, drift_est, sigma_est = model.train(TS[:N0])
    print("Estimated parameters: theta=", theta_est, " drift=", drift_est, " sigma=", sigma_est)
    model.theta = theta
    model.drift = drift
    model.sigma = sigma
    pdf_error, errors = model.probabilityfuture(TS[N0 - 1:])
    print("Probability density to observe last ", N - N0, "observations: ", pdf_error)
    print("Errors: ", errors)


if __name__ == '__main__':
    main()

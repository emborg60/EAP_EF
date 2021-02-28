from numpy import mat, cov, mean, hstack, multiply,sqrt,diag, \
    squeeze, ones, array, vstack, kron, zeros, eye, savez_compressed
from numpy.linalg import inv
from scipy.stats import chi2
from pandas import read_csv
import statsmodels.api as sm

## Work in progress ##

# Intial check from Main script
returns = dfSentPort2
riskFactors = riskFactors

# function  fmbOut = famaMacBeth(returns,riskFactors)
# Purpose:  Estimate linear asset pricing models using the Fama and MacBeth
#           (1973) two-pass cross-sectional regression methodology.
#
# Input:    returns     = TxN maxtrix of portfolio excess returns
#           riskFactors = TxK matrix of common risk factors


# Split using both named colums and ix for larger blocks
factors = riskFactors.values
portfolios = returns.values

# Use mat for easier linear algebra
factors = mat(factors)
portfolios = mat(portfolios)

# Shape information
T,K = factors.shape
T,N = portfolios.shape

excessReturns = portfolios

# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
all_coef = ts_res.params
# Cross-section regression
avgExcessReturns = mean(excessReturns, 0)
cs_res = sm.OLS(avgExcessReturns.T, sm.add_constant(beta.T)).fit()
riskPremia = cs_res.params

# Moment conditions
X = sm.add_constant(factors)
p = vstack((alpha, beta))
epsilon = excessReturns - X @ p
moments1 = kron(epsilon, ones((1, K + 1)))
moments1 = multiply(moments1, kron(ones((1, N)), X))
u = excessReturns - riskPremia[None,:] @ all_coef # beta
moments2 = u * all_coef.T # beta.T
# Score covariance
S = mat(cov(hstack((moments1, moments2)).T))
# Jacobian
G = mat(zeros((N * K + N + K, N * K + N + K)))
SigmaX = (X.T @ X) / T
G[:N * K + N, :N * K + N] = kron(eye(N), SigmaX)
G[N * K + N:, N * K + N:] = -beta @ beta.T

# Hertil
for i in range(N):
    temp = zeros((K, K + 1))
    values = mean(u[:, i]) - multiply(all_coef[:, i], riskPremia) # beta[:, i]
    temp[:, 1:] = diag(values)
    G[N * K + N:, i * (K + 1):(i + 1) * (K + 1)] = temp

vcv = inv(G.T) * S * inv(G) / T

vcvAlpha = vcv[0:N * K + N:4, 0:N * K + N:4]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(25).cdf(J)

vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB        HML')
print('--------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}     {2:0.4f}'.format(arp[0], arp[1], arp[2]))
print('Std. Err.  {0:0.4f}    {1:0.4f}     {2:0.4f}'.format(arpSE[0], arpSE[1], arpSE[2]))
print('\n\n')

print('J-test:   {:0.4f}'.format(J))
print('P-value:   {:0.4f}'.format(Jpval))

i = 0
betaSE = []
for j in range(5):
    for k in range(5):
        a = alpha[i]
        b = beta[:, i]
        variances = diag(vcv[(K + 1) * i:(K + 1) * (i + 1), (K + 1) * i:(K + 1) * (i + 1)])
        betaSE.append(sqrt(variances))
        s = sqrt(variances)
        c = hstack((a, b))
        t = c / s
        print('Size: {:}, Value:{:}   Alpha   Beta(VWM)   Beta(SMB)   Beta(HML)'.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}'.format(s[0], s[1], s[2], s[3]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}'.format(t[0], t[1], t[2], t[3]))
        print('')
        i += 1



betaSE = array(betaSE)
savez_compressed('fama-macbeth-results', alpha=alpha, beta=beta,
                 betaSE=betaSE, arpSE=arpSE, arp=arp, J=J, Jpval=Jpval)

from numpy import savez
savez('fama-macBeth-results.npz', arp=arp, beta=beta, arpSE=arpSE,
      betaSE=betaSE, J=J, Jpval=Jpval)
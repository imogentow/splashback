import numpy as np
import splashback as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import linear_model
from scipy.stats import linregress
# from sklearn import ensemble
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn import neighbors

flm = sp.flamingo("L1000N1800", "HF")
flm.read_properties()
magnitudes = np.genfromtxt(flm.path + "_galaxy_magnitudes.csv", delimiter=",")
sorted_magnitudes = np.sort(magnitudes)
mag_bcg = sorted_magnitudes[:,0]
mag_fourth = sorted_magnitudes[:,2]
flm.gap = mag_fourth - mag_bcg
N_clusters = len(flm.gap)
flm.read_2D_properties()
flm.concentration = flm.concentration[:N_clusters]
flm.alignment = flm.alignment[:N_clusters]
flm.symmetry = flm.symmetry[:N_clusters]
flm.centroid = flm.centroid[:N_clusters]

list_good = np.intersect1d(np.where(np.isfinite(flm.gap)), 
               np.intersect1d(np.where(np.isfinite(flm.M200m)),
               np.intersect1d(np.where(np.isfinite(flm.concentration)),
               np.intersect1d(np.where(np.isfinite(flm.alignment)),
               np.intersect1d(np.where(np.isfinite(flm.symmetry)),
               np.intersect1d(np.where(np.isfinite(flm.centroid)),
                              np.where(np.isfinite(flm.accretion))))))))

y = flm.accretion[list_good]
data = np.vstack((flm.gap, np.log10(flm.M200m), flm.concentration, 
                  flm.alignment, flm.symmetry, flm.centroid)).T
X = data[list_good]
X = normalize(X)

sample = np.arange(len(y))
X_train, X_test, sample_train, sample_test = train_test_split(X, sample, test_size=0.2, random_state=10)
y_train = y[sample_train]
y_test = y[sample_test]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_hat = regr.predict(X_test)

plt.scatter(y_test, y_hat)
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, xlim, color="k")
plt.xlim((0,6))
plt.ylim((0,6))
plt.xlabel("$\Gamma_{\\rm{true}}$")
plt.ylabel("$\Gamma_{\\rm{pred}}$")
plt.show()

def test(x, a, b):
    return a*x + b

def plot(value, label):
    res = linregress(value[list_good][sample_train], y_train)
    y_hat = test(value[list_good][sample_test], res.slope, res.intercept)
    
    plt.scatter(y_test, y_hat)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.plot(xlim, xlim, color="k")
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.title(label)
    plt.xlabel("$\Gamma_{\\rm{true}}$")
    plt.ylabel("$\Gamma_{\\rm{pred}}$")
    plt.show()

plot(flm.gap, "M14")

# sample = np.random.choice(np.arange(len(flm.gap)), len(y_test))
# plt.scatter(flm.gap[sample], flm.accretion[sample])
# plt.xlim((0,3))
# plt.ylim((0,6))
# plt.xlabel("M14")
# plt.ylabel("$\Gamma$")
# plt.show()

plot(flm.M200m, "$M_{\\rm{200m}}$")

# plt.scatter(flm.M200m[sample], flm.accretion[sample])
# plt.xscale('log')
# plt.xlim((1e14,1e15))
# plt.ylim((0,6))
# plt.xlabel("$M_{\\rm{200m}}$")
# plt.ylabel("$\Gamma$")
# plt.show()

plot(flm.concentration, "$c$")

# plt.scatter(flm.concentration[sample], flm.accretion[sample])
# plt.xlim((0,0.8))
# plt.ylim((0,6))
# plt.xlabel("$c$")
# plt.ylabel("$\Gamma$")
# plt.show()

plot(flm.symmetry, "$s$")

# plt.scatter(flm.symmetry[sample], flm.accretion[sample])
# plt.xlim((0,2))
# plt.ylim((0,6))
# plt.xlabel("$s$")
# plt.ylabel("$\Gamma$")
# plt.show()

plot(flm.alignment, "$a$")

# plt.scatter(flm.alignment[sample], flm.accretion[sample])
# plt.xlim((0,2))
# plt.ylim((0,6))
# plt.xlabel("$a$")
# plt.ylabel("$\Gamma$")
# plt.show()

plot(flm.centroid, "$w$")

# plt.scatter(flm.centroid[sample], flm.accretion[sample])
# plt.xlim((-3,0))
# plt.ylim((0,6))
# plt.xlabel("$w$")
# plt.ylabel("$\Gamma$")
# plt.show()
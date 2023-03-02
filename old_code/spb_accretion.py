import numpy as np
import matplotlib.pyplot as plt

PATH = "Analysis_files/Splashback/"

a_macsis = np.genfromtxt(PATH + "accretion_rates.csv", delimiter=",")
a_ceagle = np.genfromtxt(PATH + "accretion_rates_ceagle.csv", delimiter=",")

M200m_macsis = np.genfromtxt(PATH + "M_200m_macsis.csv", delimiter=",")
M200m_ceagle = np.genfromtxt(PATH + "M_200m_ceagle.csv", delimiter=",")

plt.figure()
plt.hist(a_macsis, bins=15, density=True, label="MACSIS")
plt.hist(a_ceagle, bins=15, density=True, label="C-EAGLE")
plt.legend()
plt.xlabel("$\Gamma$")
plt.show()

plt.figure()
plt.scatter(M200m_macsis*1e10, a_macsis, edgecolor="k", label="MACSIS")
plt.scatter(M200m_ceagle*1e10, a_ceagle, edgecolor="k", label="C-EAGLE")
plt.legend()
plt.xscale("log")
plt.xlabel("$M_{200m}$")
plt.ylabel("$\Gamma$")
plt.show()

print(np.corrcoef(np.hstack((M200m_macsis, M200m_ceagle)), np.hstack((a_macsis, a_ceagle))))
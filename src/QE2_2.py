import numpy as np
import quantecon as qe

# Since any of the numerical results do not depend on the choice
# of (z1, z2, z3), let's assume they're equal to (0,1,2)


# Calculating probability of transition from (xt0, zt0) to (xt1, zt1), given the matrix of transitions for Zs
def transition(state1, state2, P):
    xt0, zt0 = state1
    xt1, zt1 = state2

    xp = 0
    if zt0 == z1:
        xp = 0
    elif zt0 == z2:
        xp = xt0
    elif zt0 == z3 and xt0 <= 4:
        xp = xt0 + 1
    else:
        xp = 5

    if xt1 == xp:
        return P[zt0, zt1]
    else:
        return 0


z1 = 0
z2 = 1
z3 = 2

X = [0, 1, 2, 3, 4, 5]
Z = [z1, z2, z3]

XZ = []
for x in X:
    for z in Z:
        XZ.append((x, z))


P_Z = np.matrix([[0.5, 0.3, 0.2], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])

# Joint transition matrix
P = np.matrix([[transition(XZ[i], XZ[j], P_Z) for j in range(len(XZ))] for i in range(len(XZ))])
print('Joint transition matrix:')
print(P)

# Defining a Markov Chain
mc = qe.MarkovChain(P, XZ)

# Checking for irreducibility
irreducibility = mc.is_irreducible
print('\nP matrix irreducibility: {}'.format(irreducibility))

# Obtaining stationary distribution
stationary = np.array(mc.stationary_distributions).flatten()
print('\nStationary distribution for (X,Z) states:')
print(dict(zip(XZ, stationary)))

# Obtaining marginal distribution for x_i by summing marginal probabilities of (x_i, z_j) for j = 0, 1, 2
x_marginal = np.array([np.sum(stationary[len(Z) * i:len(Z)*(i+1)]) for i in range(len(X))])
print('\nX marginal stationary distribution:')
print(dict(zip(X, x_marginal)))

# Finding expected value
ev = np.dot(x_marginal, X)
print('\nExpected value of X: {}'.format(ev))




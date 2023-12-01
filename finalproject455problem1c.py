import numpy as np
import statistics as st

def simulation(iter):
    stoppinglst = []
    kkk = 0
    miu = 0.1
    sigma = 0.5
    R = 200
    S0 = 100
    T = 1
    N = 10000
    dt = T / N
    while kkk <= iter:
        S = np.zeros(N+1)
        t = np.zeros(N+1)
        S[0] = S0
        drift = (miu - 0.5*sigma**2)*dt
        diffusion = sigma*np.sqrt(dt)
        Z = np.random.normal(size=N)
        S = np.zeros(N+1)
        S[0] = S0
        for i in range(N):
            S[i+1] = S[i]*np.exp(drift + diffusion*Z[i])

        M = 199
        delta = R/M
        delta_t = T/N
        K = 100
        r = 0.01
        V = np.zeros((M+1, N+1))

        for i in range(M+1):
            V[i, N] = -1 * max(0, K - i*delta)
        for j in range(N+1):
            V[0, j] = -1 * max(0, K-0)
            V[M, j] = -1 * max(0, K-R)


        for j in range(N-1, -1, -1):
            for i in range(1, M):
                a = 1 - ((delta_t*(sigma**2)*((i*delta)**2))/(delta**2)) - (delta_t*miu*i*delta/delta) - delta_t*r
                b = ((delta_t*(sigma**2)*((i*delta)**2))/(2*delta**2)) + (delta_t*miu*i*delta/delta)
                c = (delta_t*(sigma**2)*((i*delta)**2))/(2*delta**2)
                V[i, j] = min(-1*max((K - i*delta), 0), a*V[i, j+1] + b*V[i+1, j+1] + c*V[i-1, j+1])

        D_mat = V.copy()
        for i in range(M+1):
            for j in range(N+1):
                D_mat[i, j] += max(0, K - i*delta)

        D_x = [] # this is x value for D boundary
        for i in range(D_mat.shape[1]):
            idx = np.argmin(D_mat[:, i] >= 0)
            D_x.append(idx*delta)

        diff = S - D_x
        indices = np.where(diff <= 0)[0]
        stoppingtime = dt * indices[0] if indices.size > 0 else -1

        if stoppingtime > 0:
            stoppinglst.append(stoppingtime)

        kkk += 1
        print("finish iteration ", kkk)

    return stoppinglst


optimaltime = simulation(100)

print("Mean is: ", st.mean(optimaltime))
print("Variance is: ", st.variance(optimaltime))

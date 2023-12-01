import numpy as np
import matplotlib.pyplot as plt

# (a) the path of stock price
miu = 0.1
sigma = 0.5
R = 200
S0 = 100
T = 1
N = 10000
dt = T/N
S = np.zeros(N+1)
t = np.zeros(N+1)
S[0] = S0
np.random.seed(12)
# for i in range(1, N+1):
#     Wt = np.random.normal(0, np.sqrt(i*dt))
#     t[i] = i*dt
#     S[i] = S[i-1]*np.exp((miu - (sigma**2/2))*dt + sigma*Wt)
drift = (miu - 0.5*sigma**2)*dt
diffusion = sigma*np.sqrt(dt)
Z = np.random.normal(size=N)
S = np.zeros(N+1)
S[0] = S0
for i in range(N):
    S[i+1] = S[i]*np.exp(drift + diffusion*Z[i])
t = np.linspace(0,T,N+1)
fig, ax = plt.subplots()
ax.plot(t,S,label='Stock Price')

# # find the value of V
M = 199  # we need to satisfy the inequality: 1-Δ*sigma^2*M^2-Δ*mu*M-Δ*r >= 0
delta = R/M
delta_t = T/N
K = 100
r = 0.01
V = np.zeros((M+1, N+1))
# boundary condition
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

# plot the boundary
D_mat = V.copy()
for i in range(M+1):
    for j in range(N+1):
        D_mat[i, j] += max(0, K - i*delta)

D_x = [] # this is x value for D boundary
for i in range(D_mat.shape[1]):
    idx = np.argmin(D_mat[:, i] >= 0)
    D_x.append(idx*delta)

print(len(S))
print(len(D_x))

D_t = np.linspace(0,T,N+1)

ax.plot(D_t[0:-1],D_x[0:-1], label='Boundary')
ax.legend()
plt.show()
# plot the second graph
t_values = [0, 0.1, 0.2, 0.5]
colors = ['r', 'g', 'b', 'k']
x = np.linspace(0, R, M+1)

plt.figure(figsize=(8, 6))
for idx, t_val in enumerate(t_values):
    n = int(t_val/delta_t)
    plt.plot(x, -1*V[:, n], colors[idx], label=f't={t_val}')

plt.xlabel('x')
plt.ylabel('V')
plt.title('Value functions -V(t,x)')
plt.legend()
plt.show()




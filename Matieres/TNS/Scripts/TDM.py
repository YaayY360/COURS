from numpy import ones, zeros, concatenate, tile, ceil, abs,arange
from tfd import tfd
from matplotlib.pyplot import figure, subplot, plot, title, xlabel, ylabel, legend, grid, show

N=200
N0=50
alpha=0.5
etats_haut=int(alpha*N0)
etats_bas=int(N0-etats_haut)
motif=concatenate((ones(etats_haut),zeros(etats_bas)))
reps=int(ceil(N/N0))
x_amp=tile(motif,reps)[:N]
X,nu=tfd(x_amp)

n=arange(len(x_amp)) #vecteur n discret
figure(figsize=(10,6))
subplot(2,1,1)
plot(n,x_amp, label='x[n]', color="blue")
title("representation en temps et en frequence du creneau")
xlabel("echantillons ")
ylabel("amplitude (V)")
grid(True)
legend()

subplot(2,1,2)
plot(nu,abs(X), label='|X(nu)|', color="red")
xlabel("frequence (Hz) ")
ylabel("module")
grid(True)
legend()
show()
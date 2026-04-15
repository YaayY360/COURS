# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
sig_f0=1000
sig_fe=44100
sig_nu0=sig_f0/sig_fe
sig_duree=5
sig_n=np.arange(sig_duree*sig_fe)   ##on crée un vecteur d'indice
sig_amp=np.sin(2*np.pi*sig_nu0*sig_n)  ##création de la sinusoide

v_reduc=0.001
moitie=len(sig_n)//2 ##on divise par deux le signal attention il faut mettre une double "//"
tab1=np.ones(moitie)    ##on affecte la valeur 1 à une moitié
tab2=np.ones(moitie)*v_reduc   ##on affecte l'autre moitié à la valeur 0.1
amp=np.concatenate((tab1,tab2))
sig1=sig_amp*amp

duree_blanc=0.005
n_blanc=int (duree_blanc*sig_fe)
n_cote=(len(sig_n)-n_blanc)//2

partie_1=np.ones(n_cote)
partie_silence = np.zeros(n_blanc)
partie_2=np.ones(len(sig_n)-n_cote-n_blanc)
amp2=np.concatenate((partie_1,partie_silence,partie_2))
sig2=sig_amp*amp2

deltaf=50
part1=np.ones(moitie)*sig_f0
part2=np.ones(moitie)*(sig_f0+deltaf)
amp3=np.concatenate((part1,part2))
sig3=np.sin(2*np.pi*(amp3/sig_fe)*sig_n)

attenuation_dB=20*np.log10(v_reduc)
print("L'atténuation est de {} dB".format(attenuation_dB))

#plt.plot(sig_n,sig_amp)
plt.stem(sig_n[:150],sig_amp[:150])
# plt.plot(sig_n[:500],sig_amp[:500])
plt.title("Test d'audition")
plt.xlabel("Echantillon (n)")
plt.ylabel("Amplitude [v]")
plt.grid(True)
plt.show()

sd.play(sig1,sig_fe)
sd.wait()
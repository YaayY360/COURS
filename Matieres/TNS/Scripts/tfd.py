from scipy.fft import fft, fftshift, fftfreq
def tfd(seq):
    Nfft=len(seq)
    SEQ=fft(seq)
    SEQ_centree=fftshift(SEQ)
    nu=fftshift((fftfreq(Nfft)))
    return SEQ_centree,nu

def tfd_noshift(seq):
    Nfft=len(seq)
    SEQ=fft(seq)
    nu=fftshift((fftfreq(Nfft)))
    return SEQ,nu
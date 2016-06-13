def fr_spectrum(s1,sampFreq):
        from scipy.fftpack import fft
        from math import ceil
        from numpy import log10
        n = len(s1)
        p = fft(s1) # take the fourier transform
        nUniquePts = ceil((n+1)/2.0)
        p = p[0:nUniquePts]
        p = abs(p)
        p = p / float(n)
        p = p**2
        if n % 2 > 0: # we've got odd number of points fft
                p[1:len(p)] = p[1:len(p)] * 2
        else:
                p[1:len(p) -1] = p[1:len(p) - 1] * 2
        freqArray = plt.arange(0, nUniquePts, 1.0) * (sampFreq / n);
        plt.plot(freqArray/1000, 10*log10(p), color='k')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power (dB)')
        plt.show()


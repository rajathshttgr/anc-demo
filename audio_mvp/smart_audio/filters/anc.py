
import numpy as np

class LMSANC:
    """Feedforward ANC using LMS adaptive filter.
    x: reference noise (correlated with disturbance)
    d: primary signal (speech + noise)
    Output is error e = d - y where y = w^T x_vec
    """
    def __init__(self, filter_len=128, mu=0.001):
        self.L = int(filter_len)
        self.mu = float(mu)
        self.w = np.zeros(self.L, dtype=np.float32)
        self.xbuf = np.zeros(self.L, dtype=np.float32)

    def process(self, x, d):
        assert len(x) == len(d)
        y_out = np.zeros_like(d, dtype=np.float32)
        e_out = np.zeros_like(d, dtype=np.float32)
        for n in range(len(d)):
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = x[n]
            y = np.dot(self.w, self.xbuf)
            e = d[n] - y
            # LMS update
            self.w += 2*self.mu*e*self.xbuf
            y_out[n] = y
            e_out[n] = e
        return e_out, y_out

class NLMSANC:
    """Normalized LMS improves stability wrt input power."""
    def __init__(self, filter_len=128, mu=0.5, eps=1e-6):
        self.L = int(filter_len)
        self.mu = float(mu)
        self.eps = float(eps)
        self.w = np.zeros(self.L, dtype=np.float32)
        self.xbuf = np.zeros(self.L, dtype=np.float32)

    def process(self, x, d):
        assert len(x) == len(d)
        y_out = np.zeros_like(d, dtype=np.float32)
        e_out = np.zeros_like(d, dtype=np.float32)
        for n in range(len(d)):
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = x[n]
            y = np.dot(self.w, self.xbuf)
            e = d[n] - y
            norm = np.dot(self.xbuf, self.xbuf) + self.eps
            self.w += (self.mu / norm) * e * self.xbuf
            y_out[n] = y
            e_out[n] = e
        return e_out, y_out

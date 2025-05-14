# Definition of the DFE models
# Probablity density function for the DFE

def neugamma(xx, params):
    """Return Neutral + Gamma distribution"""
    """params = (shape, scale, pneu)
    pneu is the proportion of neutral mutations"""
    alpha, beta, pneu = params
    xx = np.atleast_1d(xx)
    out = (1-pneu)*ssd.gamma.pdf(xx, alpha, scale=beta)
    # Assume gamma < 1e-4 is essentially neutral
    out[np.logical_and(0 <= xx, xx < 1e-4)] += pneu/1e-4
    # Reduce xx back to scalar if it's possible
    return np.squeeze(out)


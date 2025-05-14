# dadi-cli_neugamma
# Add neugamma distribution to dadi-cli.
# 0. Define neu-gamma distribiton
def neugamma(xx, params):
   """Return Neutral + Gamma distribution"""
   """params = (shape, scale, pneu)
   pneu is the proportion of neutral mutations"""
   alpha, beta, pneu = params
   xx = np.atleast_1d(xx)
   out = (1-pneu)*gamma(xx, (alpha, beta))
   # Assume gamma < 1e-4 is essentially neutral
   out[np.logical_and(0 <= xx, xx < 1e-4)] += pneu/1e-4
   # Reduce xx back to scalar if it's possible
   return np.squeeze(out)

# 1. Add neugamma distribution to [path to conda environment or path_to the package dadi]/lib/python3.10/site-packages/dadi/DFE/PDFs.py

# 2. Add neugamma option to [path to conda environment or path_to the package dadi]/lib/python3.10/site-packages/dadi_cli/Pdfs.py


# 3. The dadi-cli bash command to generate cache is unchanged.
# 4. Add pneu to the parameters of DFE inference.
pinitials=$shape $scale $pneu
lbounds=$lshape $lscale $lpneu
ubounds=$ushape $uscale $upneu
dadi-cli InferDFE --fs file.fs --cache1d cache.spectra.bpkl --pdf1d neugamma --p0 $pinitials --lbounds $lbounds --ubounds $ubounds --demo-popt ...




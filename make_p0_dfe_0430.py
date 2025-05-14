# 2025/03/17
# Add an option of adjusting mutation rate by adding two new parameters to make_p0_dfe_0430.py
# 1. muadj_ctrl
# 2. muadj_test
# 2025/02/27
# Add an option of input mutation_ctrl_diploid and mutation_test_diploid to avoid reading fsmut files.
# 2024/11/19
# Add pdf=neugammazero
# 2024/04/29
# Modified the naming strategy of the input files to match the updated file names.
#make_p0_dfe.py
#Use state, demographic model, DFE PDF, 
# 2024/02/28
# Add an input argument "state" to be complementary to "annofile". The input is directly the state name used for specifying fs and mutation.
# The maxll_demoparam_file was modified to have headings. I changed to match "state" in col[1].
# 2024 Jan 30
#demoparamaerke that decide the directory of demo inference file, cachemarker decide cache file, 
#dfeparammarer to track the input parameters for dfe inference.
import numpy as np
from itertools import product
import os
import subprocess
import argparse
import csv
import shutil


def main(args):
    # Your main script logic here
    print("dirout: {}".format(args.dirout))
    print("testfs: {}".format(args.testfs))
    print("cachefile: {}".format(args.cachefile))
    print("demopopt: {}".format(args.demopopt))
    print("pdf: {}".format(args.pdf))
    print("mutation_ctrl_diploid: {}".format(args.mutation_ctrl_diploid))
    print("mutation_test_diploid: {}".format(args.mutation_test_diploid))
    print("fsmutc: {}".format(args.fsmutc))
    print("fsmutt: {}".format(args.fsmutt))
    # coefficients used to adjust mutation rate for control and test
    print("muadj_ctrl: {}".format(args.muadj_ctrl))
    print("muadj_test: {}".format(args.muadj_test))

    print("Na: {}".format(args.Na))
    print("lrange_dfe: {}".format(args.lrange_dfe))
    print("urange_dfe: {}".format(args.urange_dfe))
    print("p0lrange_dfe: {}".format(args.p0lrange_dfe))
    print("p0urange_dfe: {}".format(args.p0urange_dfe))
    print("num_bins: {}".format(args.num_bins))
    print("optcount: {}".format(args.optcount))
    print("maxeval: {}".format(args.maxeval))
    print("num_cpu: {}".format(args.num_cpu))
    print("num_convergence: {}".format(args.num_convergence))
    print("packnumber: {}".format(args.packnumber))


if __name__ == "__main__":
    # Create argument parser 
    parser = argparse.ArgumentParser(description="Your script description")

    # Add arguments with flags, descriptions, and default values
    parser.add_argument("-dirout", "--dirout", type=str, help="directory of output files")
    parser.add_argument("-testfs", "--testfs", type=str, help="FS file for fittng DFE.")
    parser.add_argument("-cachefile", "--cachefile", type=str, help="Cache file based on the control demographic inference.")
    parser.add_argument("-demopopt", "--demopopt", type=str, help="demographic parameters inference summary file from dadi-cli.")
    parser.add_argument("-pdf", "--pdf", type=str, help="Probability density function for DFE. ['gamma','neugamma','neugammazero']", default='gamma')
    parser.add_argument("-fsmutc", "--fsmutc", type=str, help="fsmut file for controls that used to infer demographic model.")
    parser.add_argument("-fsmutt", "--fsmutt", type=str, help="fsmut file for test that we want to infer DFE.")
    parser.add_argument("-mutnumdipc", "--mutation_ctrl_diploid", type=float, help="Mutation rate for control population. If not provided, it will be calculated from fsmut file.", default=None)
    parser.add_argument("-mutnumdipt", "--mutation_test_diploid", type=float, help="Mutation rate for test population. If not provided, it will be calculated from fsmut file.", default=None)
    parser.add_argument("-muadj_ctrl", "--muadj_ctrl", type=float, help="Coefficient used to adjust mutation rate for control population.", default=None)
    parser.add_argument("-muadj_test", "--muadj_test", type=float, help="Coefficient used to adjust mutation rate for test population.", default=None)
    #parser.add_argument("-dfemarker", "--dfemarker", type=str, help="A number used as a marker for parameters used for generating the set of parameters for inferring DFE.", default='1')

    parser.add_argument("-Na", "--Na", type=str, help="Ancestral population size inferred from demographic model:= theta/(4ul)")
    # If Na is not calculated. Get it from the best demographic inference and fsmut file.    
    parser.add_argument("-lr", "--lrange_dfe", type=str, help="Lower range of dfe_params. If pdf='gamma', dfe_params=[shape, scale]. If pdf='neugamma', check how you define the function. It may be [shape,scale,pneu] or [pneu, shape, scale] ", default='1e-10,1e-10')
    parser.add_argument("-ur", "--urange_dfe", type=str, help="Upper range of dfe_params. If pdf='gamma', dfe_params=[shape, scale]. If pdf='neugamma', check how you define the function. It may be [shape,scale,pneu] or [pneu, shape, scale] ", default='2,1e8')
    parser.add_argument("-p0lr", "--p0lrange_dfe", type=str, help="The lower range of the initial parameters.", default=parser.parse_known_args()[0].lrange_dfe)
    parser.add_argument("-p0ur", "--p0urange_dfe", type=str, help="The upper range of the initial parameters.", default=parser.parse_known_args()[0].urange_dfe)
    parser.add_argument("-numb", "--num_bins", type=int, help="The number of values within the range of p0 of each paramenter used for making a combination of p0.", default=30)
    
    parser.add_argument("-optcount", "--optcount", type=int, help="Number of optimizations.", default=100)
    parser.add_argument("-maxeval", "--maxeval", type=int, help="Maximum number of evaluations.", default=200)
    parser.add_argument("-num_cpu", "--num_cpu", type=int, help="Number of CPUs.", default=10)
    parser.add_argument("-num_convergence", "--num_convergence", type=int, help="Number of convergence.", default=5)

    parser.add_argument("-packnumber", "--packnumber", type=int, help="Number of command lines in each bash file.", default=100)
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)


# get arguments 
dirout=args.dirout
testfs=args.testfs
cachefile=args.cachefile
demopopt=args.demopopt
pdf=args.pdf
fsmutc=args.fsmutc
fsmutt=args.fsmutt
mutation_ctrl_diploid=args.mutation_ctrl_diploid
mutation_test_diploid=args.mutation_test_diploid
muadj_ctrl=args.muadj_ctrl
muadj_test=args.muadj_test
Na=float(args.Na)
lrange_dfe=args.lrange_dfe
urange_dfe=args.urange_dfe
p0lrange_dfe=args.p0lrange_dfe
p0urange_dfe=args.p0urange_dfe
num_bins=args.num_bins
optcount=args.optcount
maxeval=args.maxeval
num_cpu=args.num_cpu
num_convergence=args.num_convergence
packnumber=args.packnumber

# test parameters
# rootdir='/u/home/c/cdi/project-klohmuel/noncdoingdfe/test_0118'
# annofile='enhancers_states_hg38'
# model="three_epoch"
# demoparammarker='1'
# cachemarker='1'
# dfeparammarker='1'

# pdf='gamma'
# pop='YRI'
# optcount=100
# maxeval=200
# num_cpu=10
# num_convergence=5

# lrange_dfe='1e-10,1e-10'
# urange_dfe='2,1e8'
# p0lrange_dfe='1e-10,1e-10'
# p0urange_dfe='2,1e8'
# num_bins=30
# mutfile=rootdir+'/output_mut/roulette_mutul.enhancers_states_hg38.tab'



# functions 
def valid_p0s (pdf ,p0lrange_dfe, p0urange_dfe,gamma0_range,num_bins):
    '''
    Description:
        Generate valid pairs initial parameters for dadi-cli InferDFE.
    Arguments:
        pdf string: Name of the probability density function.
        p0lrange_dfe list: Lower range of initial parameters.
        p0urange_dfe list: Upper range of initial parameters.
        gamma0_range list: Range of gamma0.
            suggestion: gammal, gammau=2*0.000001*Na, 2*0.5*Na
                        gamma0_range=[gammal, gammau]
        num_bins int: Number of bins for each parameter.
    '''
    shape0s = np.logspace(np.log10(p0lrange_dfe[0]), np.log10(p0urange_dfe[0]), num=num_bins)
    scale0s = np.logspace(np.log10(p0lrange_dfe[1]), np.log10(p0urange_dfe[1]), num=num_bins)
    # Generate pairs and filter based on the product falling within gamma0_range
    # Add the combination of pneu if it's neugamma
    if pdf =="neugamma" or pdf =="neugammazero":
        pneus = np.linspace(p0lrange_dfe[2], p0urange_dfe[2], num=num_bins)
        valid_pairs = [(shape0, scale0, pneu) for shape0, scale0, pneu in product(shape0s, scale0s, pneus) if gamma0_range[0] <= shape0 * scale0 <= gamma0_range[1]]
    elif pdf =="gamma":
        valid_pairs = [(shape0, scale0) for shape0 in shape0s for scale0 in scale0s if gamma0_range[0] <= shape0 * scale0 <= gamma0_range[1]]
    else:
        print("PDF is not gamma, neugamma or neugammazero. Please include the custermized PDF in valid_p0s.")
    return valid_pairs

def getul(fsmutfile):
    roulette_scale=1.015e-7
    with open(fsmutfile, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        # Skip the header
        # Get the first row after the header
        first_row_after_header = next(reader)
        # check if the string fsfile column contain "etal"
        if "etal" in first_row_after_header['fsfile']:
            #print(first_row_after_header['fsfile'])
            # Get the first row after the header
            l = float(first_row_after_header['length_adj_inputbed'])
            # Calculate the mutation number for each chromosome. ul=hapmutrate*L_mutational
            u = roulette_scale * float(first_row_after_header['avepersite_diploi_Roulette_mutrate']) / 2
            ul=u*l
        else:
            print("Format Error in mutfile: The fsfile column does not contain 'etal'.")
    mutation_ctrl_diploid=2*float(ul)
    return mutation_ctrl_diploid

def get_opts_and_theta(filename, gen_cache=False):
    """
    Description:
        Obtains optimized parameters and theta.

    Arguments:
        filename str: Name of the file.
        gen_cache bool: Make True for generating a cache to remove misid parameter when present.

    Returns:
        opts list: Optimized parameters.
        theta float: Population-scaled mutation rate.
    """
    opts = []
    param_names = []
    is_converged = False
    with open(filename, "r") as fid:
        for line in fid.readlines():
            if line.startswith("# Converged"):
                is_converged = True
                continue
            elif line.startswith("# Log(likelihood)"):
                param_names = line.rstrip().split("\t")
                continue
            elif line.startswith("#"):
                continue
            else:
                try:
                    opts = [float(_) for _ in line.rstrip().split("\t")]
                    break
                except ValueError:
                    pass
    theta = opts[-1] 
    if gen_cache and "misid" in param_names:
        opts = opts[1:-2]
    else:
        opts = opts[1:-1]
    if not is_converged:
        print("No converged optimization results found.")
    return opts, theta

# Get mutation ratio between control and test
# Get mutation rate for control
if mutation_ctrl_diploid is None:
    mutation_ctrl_diploid=getul(fsmutc)
# Get mutation rate for test
if mutation_test_diploid is None:
    mutation_test_diploid=getul(fsmutt)

# Adjust mutation rate for test to control, the default is 1.
# Calculate the ratio of mutation rate for test to control
# if mutadj_ctrl is not None, adjust mutation rate for control by multiplying it by mutadj_ctrl
if muadj_ctrl is not None:
    mutation_ctrl_diploid=mutation_ctrl_diploid*muadj_ctrl
# if mutadj_test is not None, adjust mutation rate for test by multiplying it by mutadj_test
if muadj_test is not None:
    mutation_test_diploid=mutation_test_diploid*muadj_test
    
mutratio=mutation_test_diploid/mutation_ctrl_diploid


## Get range of parameters
lrange_dfe = [float(element) for element in lrange_dfe.split(",")]
urange_dfe = [float(element) for element in urange_dfe.split(",")]
p0lrange_dfe = [float(element) for element in p0lrange_dfe.split(",")]
p0urange_dfe = [float(element) for element in p0urange_dfe.split(",")]


# define the range of gamma by Na and add gamma to params_dfe
## If Na is not in input, calculate it from the best demographic inference and fsmut file.
if Na is None:
    # Get the theta and opts from the best demographic inference
    opts_demog, theta_demog = get_opts_and_theta(demopopt)
    # Calculate Na from theta and mutation rate
    Na = theta_demog / (2 * mutation_ctrl_diploid)

gammal, gammau=2*0.000001*Na, 2*0.5*Na
gamma0_range=[gammal, gammau]

# make pairs of shape0 and scale0 and only keep whose proudct is in gamma0_range
valid_pairs=valid_p0s(pdf,p0lrange_dfe, p0urange_dfe,gamma0_range,num_bins)
num_tests=len(valid_pairs)
print("Number of valid pairs of initial parameters: "+str(num_tests))

# Convert the elements of lrange_dfe to strings and join them with spaces
lrange_str = ' '.join(map(str, lrange_dfe))
# Convert the elements of urange_dfe to strings and join them with spaces
urange_str = ' '.join(map(str, urange_dfe))

# make directory if not exist
fulldirdfe=dirout
if not os.path.exists(fulldirdfe):
    os.makedirs(fulldirdfe)
    print("Directory ", fulldirdfe, " Created ")
else:
    print("Directory ", fulldirdfe, " already exists")
    # rename the original directory
    # if .bak directory exists, remove it
    if os.path.exists(fulldirdfe+'.bak'):
        shutil.rmtree(fulldirdfe+'.bak')
        print(fulldirdfe+".bak exists, remove it")
    shutil.move(fulldirdfe, fulldirdfe+'.bak')
    print(fulldirdfe+" file exists, rename it to "+fulldirdfe+".bak")
    # create a new directory
    os.makedirs(fulldirdfe)
    print("Directory ", fulldirdfe, " Created ")

os.chdir(fulldirdfe)
# Write bash command loop through dfemarkers
dfemarkers =list(map(int, np.linspace(0, num_tests-1, num=num_tests)))

for dfemarker in dfemarkers:
    i=dfemarker
    print(dfemarker)
    p0_dfe=' '.join(map(str, valid_pairs[dfemarker]))
    outputfile=dirout+"/"+str(i)
    line='dadi-cli InferDFE --fs '+str(testfs)+' --cache1d '+cachefile+' --pdf1d '+pdf+' --p0 '+p0_dfe+' --lbounds '+lrange_str+' --ubounds '+urange_str +' --demo-popt '+demopopt+' --ratio '+str(mutratio)+' --output-prefix '+outputfile+' --optimizations '+str(optcount)+' --maxeval '+str(maxeval)+' --nomisid --cpu '+str(num_cpu)+' --check-convergence '+str(num_convergence)
    #print(line)
    # if i residue of 1000 is 1, write the next 1000 lines to a new file
    if i%packnumber==0:
        output='dadi-cli_dfe'+'.'+str(i)+'.sh'
        if os.path.exists(output):
            os.rename(output, output+'.bak')
            print(output+" file exists, rename it to "+output+".bak")
        else:
            print(output+" file does not exist, create a new one")
        # write headers to output
    with open(output, 'a') as f:
        f.write(line+"\n")
        f.close()



# Record the strategies used in generating input parameters for dfe inference
file_record="makefile_p0_dfeinput.tab"
with open(fulldirdfe+"/"+file_record, 'a') as f:
    #if file does not exist write header
    if os.stat(fulldirdfe+"/"+file_record).st_size == 0:
        header='dfeparammarker'+'\t'+'state\tmutratio'+'\t'+'model_demog'+'\t'+'cachefile'+'\t'+'p0lrange_dfe'+'\t'+'p0urange_dfe'+'\t'+'lrange_dfe'+'\t'+'urange_dfe'
        f.write(header+'\n')
    line=fulldirdfe+'\t'+str(mutratio)+'\t'+demopopt+'\t'+cachefile+'\t'', '.join(map(str, p0lrange_dfe))+'\t'+', '.join(map(str, p0urange_dfe))+'\t'+', '.join(map(str, lrange_dfe))+'\t'+', '.join(map(str, urange_dfe))
    f.write(line+'\n')
    f.close()

print("The input parameters for dfe inference are generated and saved in "+fulldirdfe+"/"+file_record)
print("The bash files for dfe inference are generated and saved in "+fulldirdfe)





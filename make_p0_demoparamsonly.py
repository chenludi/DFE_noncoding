# 2025/02/20
# Add an argument -eskip, --edge_skip to skip the edge of the defined range to make the combination of initial params.
# This will avoid starting from extreme values of the range. However, you can also use a different p0lrange_demo and p0urange_demo to avoid this.
# This is helpful for a small num_points that may not cover the range well. 
# Add space_mode argument to specify space the range linearly or logarithmically. 
# Currently only has option of 'log' or 'linear'. Default is 'linear'. and if want 'log', specify by -smode log.
# 2024/04/21
# Make general p0 files that doesn't have fs file information. This can be reused for other runs.
# Modify to fit dadi-cli_demog_fsfile.sh: replace state to the fsfilebasename, e.g. the argument for fsfilebasename is "nc_enhancers_states_hg38" and the fs file is "nc_enhancers_states_hg38.fs"
# Remove the mutation argument because demographic inference doesn't need mutation file.
# makefile_p0
# 20240225, Chenlu Di
# Add an input argument "state" to be complementary to "annofile". The input is directly the state name used for specifying fs and mutation.
# 20240127, Chenlu Di
# Make p0 file for three epoch demographic model using original annotated filename as input.
# model_demog="three_epoch"

import numpy as np
from itertools import product
import os
import sys
import os
import argparse
import csv

def main(args):
    # Your main script logic here
    print("outputdir: {}".format(args.outputdir))
    print("model: {}".format(args.model))
    print("parammarker: {}".format(args.parammarker))
    print("lrange_demo: {}".format(args.lrange_demo))
    print("urange_demo: {}".format(args.urange_demo))
    print("p0lrange_demo: {}".format(args.p0lrange_demo))
    print("p0urange_demo: {}".format(args.p0urange_demo))
    print("num_points: {}".format(args.num_points))
    print("edge_skip: {}".format(args.edge_skip))
    print("space_mode: {}".format(args.space_mode))
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Your script description")

    # Add arguments with flags, descriptions, and default values
    parser.add_argument("-odir", "--outputdir", type=str, help="Output directory for calling files.", default='/u/home/c/cdi/project-klohmuel/noncdoingdfe/test_0118')
    parser.add_argument("-m", "--model", type=str, help="Demographic model:three_epoch,two_epoch,growth,bottlegrowth_1d", default="three_epoch")
    parser.add_argument("-pmarker", "--parammarker", type=str, help="A number used as a marker for parameters used for generating the set of parameters for inferring demographic model.", default='1')
    parser.add_argument("-lr", "--lrange_demo", type=str, help="The lower range of demographic parameters. Growth: (nu,T), Two epoch: (nu,T), Three epoch: (nuB,nuF,TB,TF), Bottlegrowth_1d: (nuB,nuF,T).  ", default='1,2,0.1,1e-6')
    parser.add_argument("-ur", "--urange_demo", type=str, help="The upper range of demographic parameters.", default='10,1000,1,0.5')
    parser.add_argument("-p0lr", "--p0lrange_demo", type=str, help="The lower range of the initial parameters.", default=parser.parse_known_args()[0].lrange_demo)
    parser.add_argument("-p0ur", "--p0urange_demo", type=str, help="The upper range of the initial parameters.", default=parser.parse_known_args()[0].urange_demo)
    parser.add_argument("-nump", "--num_points", type=int, help="The number of values within the range of p0 of each paramenter used for making a combination of p0.", default=8)
    parser.add_argument("-eskip", "--edge_skip", type=bool, help="Skip the edge of the defined range to make the combination of initial params.", default=False)
    parser.add_argument("-smode", "--space_mode", type=str, help="The mode of space for generating the set of parameters for inferring demographic model.", default='linear')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)


# get arguments 
outputdir=args.outputdir
model=args.model
parammarker=args.parammarker
lrange_demo=args.lrange_demo
urange_demo=args.urange_demo
p0lrange_demo=args.p0lrange_demo
p0urange_demo=args.p0urange_demo
num_points=args.num_points
edge_skip=args.edge_skip
space_mode=args.space_mode


# convert arguments to variables
model_demog=model
lrange_demo = [float(element) for element in lrange_demo.split(",")]
urange_demo = [float(element) for element in urange_demo.split(",")]
p0lrange_demo = [float(element) for element in p0lrange_demo.split(",")]
p0urange_demo = [float(element) for element in p0urange_demo.split(",")]

recfile=outputdir+"/record_makefile_p0_demog_input.tab"

# Check if the file does not exist or is empty
if not os.path.isfile(recfile) or os.stat(recfile).st_size == 0:
    # Define the header with tab-separated values
    header = 'model_demog\tparammarker\tp0lrange_demo\tp0urange_demo\tlrange_demo\turange_demo'

    # Open the file in 'a' mode to append or create if it doesn't exist
    with open(recfile, 'a') as f:
        # Write the header to the file
        f.write(header + '\n')
        f.close()

with open(recfile, 'a') as f:

    line=model_demog+'\t'+parammarker+'\t'+', '.join(map(str, p0lrange_demo))+'\t'+', '.join(map(str, p0urange_demo))+'\t'+', '.join(map(str, lrange_demo))+'\t'+', '.join(map(str, urange_demo))
    f.write(line+'\n')
    f.close()

# Generate evenly spaced values for each parameter
# It's better to skip the edge of defined range:
# options of space by log
if space_mode=='log':
    if edge_skip:
        # Generate evenly spaced values for each parameter, skipping the endpoints
        param_values = [np.logspace(np.log10(l), np.log10(u), num_points + 2)[1:-1] for l, u in zip(p0lrange_demo, p0urange_demo)]
    else:
        param_values = [np.logspace(np.log10(l), np.log10(u), num_points) for l, u in zip(p0lrange_demo, p0urange_demo)]
else:
    if edge_skip:
        # Generate evenly spaced values for each parameter, skipping the endpoints
        param_values = [np.linspace(l, u, num_points + 2)[1:-1] for l, u in zip(p0lrange_demo, p0urange_demo)]
    else:
        param_values = [np.linspace(l, u, num_points) for l, u in zip(p0lrange_demo, p0urange_demo)]


# remove duplicates in param_values
param_values = [np.unique(x) for x in param_values] 

# Generate all combinations of parameter values
combinations = list(product(*param_values))
# Make lines of p0_3e.txt file

# Convert the elements of lrange_demo to strings and join them with spaces
lrange_str = ' '.join(map(str, lrange_demo))
# Convert the elements of urange_demo to strings and join them with spaces
urange_str = ' '.join(map(str, urange_demo))

# make directory if not exist
runmarker=model_demog+'_'+parammarker
dirrunmarker=outputdir+'/'+model_demog+'_'+parammarker
if not os.path.exists(dirrunmarker):
    os.makedirs(dirrunmarker)
    print("Directory ", dirrunmarker, " Created ")
else:
    print("Directory ", dirrunmarker, " already exists")
    # rename the original directory
    os.rename(dirrunmarker, dirrunmarker+'.bak')
    print(dirrunmarker+" file exists, rename it to "+dirrunmarker+".bak")
    # create a new directory
    os.makedirs(dirrunmarker)
    print("Directory ", dirrunmarker, " Created ")

os.chdir(dirrunmarker)

# write parameters to output
# Print the combinations
for i in range(len(combinations)):
        combo=combinations[i]
        testmarker=str(i)
        print(i)
        # if i residue of 1000 is 1, write the next 1000 lines to a new file
        if i%100==0:
            output='p0_'+model_demog+'_'+parammarker+'_'+str(i)+'.txt'
            if os.path.exists(output):
                os.rename(output, output+'.bak')
                print(output+" file exists, rename it to "+output+".bak")
            else:
                print(output+" file does not exist, create a new one")
            # write headers to output
            #with open(output, 'a') as f:
                #f.write(line1)
                #f.write(line2)
                #f.close()
        #combo_str = ' '.join(map(lambda x: f'{x:.2f}', combo))
        combo_str = ' '.join(map(str, combo))
        line=testmarker+" "+combo_str+" "+lrange_str+" "+urange_str
        with open(output, 'a') as f:
            f.write(line+'\n')
            f.close()



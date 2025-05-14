# 2025/02/05 Chenlu Di
# A script used to get 1kgp vcf file with biallelic SNPs for given samples.
# The script runs for each chromosomes to speed up the process.
# I used h_data=16G, it might take more than 10 hours for some chromosomes.

# The input arguments are
# inputvcf, inputsample,maskfile,chr,dirout
# specify the input arguments with tags
while getopts c:v:s:m:o: flag
do
    case "${flag}" in
        c) chr=${OPTARG};;
        v) inputvcf=${OPTARG};;
        s) inputsample=${OPTARG};;
        # input sample is a txt file that each line is one sample name.
        # e.g. 
        # NA18488
        # NA19222
        # NA18867
        m) maskfile=${OPTARG};;
        o) dirout=${OPTARG};;
    esac
done

## example of input will be
## 0. inputs and outputs
## inputs: 1kg30X unphased vcf file; sample name.
## Specifying arguments: chr=21
# chr='chr21'

## Specifiy the input directories and files
# dirvcf=/u/project/klohmuel/DataRepository/NYGC_1KG/genotype

## Specify the input vcf file files
# inputvcf=$dirvcf"/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_"$chr".recalibrated_variants.vcf.gz"

## Put together the sub-pop samples.
## The population samples can be found from 1KGP or locally at:
## for indenpendent samples: /u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/popfile_2504.txt 
## Put the samples together to make your own population. 
## Super popualtion code description: https://useast.ensembl.org/Help/Faq?id=532

# inputsample='/u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/samples_YRI_2504_108.txt' 

## Use high quality sites in the mask file. Please find more info in the README file in the same directory. 
# maskfile='/u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/originalfile/20160622.allChr.mask.bed' # default mask file
# dirout='/u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/' # change to your won output directory.

#****# Usage:
# ./uti_human_snps_SFS.sh -c chr21 -v $inputvcf -s $inputsample -m $maskfile -o $dirout

# The important output is bi-allelic SNPs in the input sample file and masked by the mask file.
# It is named after the input sample file name and chromosome number. [mask.bisnp.'$inputsample_fn".1kGP."$chr".vcf.gz"]
# You can skip the last step of counting length if you are confident that everything is correct. 


## After this, you can easiy use dadi to get the SFS by:
# python
# import dadi
# inputvcf=[/u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/mask.bisnp.YRI_2504_108.1kGP.chr21.vcf.gz]
# inputpopfile=[/u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/popfile_2504_YRI108.txt]
## The inputpopfile for dadi is different from the input in this script.
## The inputpopfile for dadi is a text file that each line is [samplename]\t[popname], 
## e.g. 
## NA18488^IYRI$
## NA19222^IYRI$
# dd = dadi.Misc.make_data_dict_vcf(inputvcf, inputpopfile)
## make frequency spectrum. 
## e.g. inputpopname="YRI"
## e.g. projectnum=160
# fs = dadi.Spectrum.from_data_dict(dd, [inputpopname], projections = [projectnum], polarized = False)
# To make sure things are correct, you can count the number of sites input vcf file and sites in SFS. The number should be close but not same.

echo "Step 0: Specify inputs and outputs"
# Specify the chromosome number. Different chromosomes take different time to run because of the size variation.
#chr='chr21'
echo "Working on "$chr
#specify output directories and files
## output file
# dirout='/u/home/c/cdi/project-klohmuel/noncdoingdfe/data/vcf_1kg220425/'
# if dirout does not end with '/', add '/'
if [[ $dirout != */ ]]; then
    dirout=$dirout/
fi
## temporary directory
dirtemp=$dirout'temp/'
mkdir $dirtemp

## simplify the input names to write output names
inputsample_fn=$(basename "$inputsample")
inputsample_fn=${inputsample_fn%.*}
inputsample_fn="${inputsample_fn#*_}"

### Name the output file by samples and chr.
output1=$dirtemp$inputsample_fn".1kGP."$chr".vcf.gz" # vcf file for given samples
output2=$dirout'poly.'$input2_name # biallelic vcf file
output3=$dirout'mask.bisnp.'$inputsample_fn".1kGP."$chr".vcf.gz" # pass mask file to get high quality sites 
output5=$dirout'csites.'$chr.tab # record the number of sites in different steps

# 1. Get whole chr biallelic unmasked vcf for dadi
# 1.1 only Keep YRI samples in VCF file to speed up
echo "Step 1.1 only Keep YRI samples in VCF file to speed up"
## Use bcftools to get samples in inputsample file.
echo "Load bcftools"
module load bcftools
echo "time bcftools view -S $inputsample $dirvcf$inputvcf --force-samples | gzip -c > $output1"
echo "This step takes a long time. Make sure to allocate enough time and memory. For reference, chr21 took 30-50 minutes, tested in using 10G memory but could work using less memory."
time bcftools view -S $inputsample $dirvcf$inputvcf --force-samples | gzip -c > $output1
## Check if the process has finished correctly.
echo "Check if the process has finished correctly."
## The lines of output1 should be same or very clsose to the input.
## If the difference is large, kill the script.
### Count the number of lines in inputvcf and output1.
lines_inputvcf=$(zcat $dirvcf$inputvcf | wc -l)
lines_output1=$(zcat $output1 | wc -l)
### Calculate the absolute difference in line counts
diff_lines=$((lines_inputvcf - lines_output1))
absolute_diff_lines=${diff_lines#-}
### Check if the absolute difference is greater than 10
if [ "$absolute_diff_lines" -gt 10 ]; then
    echo "Error: erros in step1 in getting samples. The difference in line counts between 1kgp input vcf file and output sample.vcf file is more than 10 lines. Maybe increasing time and memory.Exiting."
    exit 1
fi
###output file is YRI_2504_108.1kGP.chr21.vcf.gz

# 1.2 Only keep biallelic SNPs by using bcftools
echo "Step 1.2 Only keep biallelic SNPs by using bcftools.This took 6-10 min for chr21."
input2=$output1
input2_name=$(basename "$input2")
# The first two lines of bcftools commands are not necessay for getting biallelic SNPs but can be used to check data.
# I keep these unnecessary lines for tracking the number of different kinds of variants.
## Remove monomorphic sites in the population
### I think if a site will be considered as monomorphic if it is 0 or 1 at all samples except for missing samples. That is only having '0' and '.' is monomorphic, same as '1' and '.'
echo "Remove monomorphic sites in the population."
bcftools view -c 1 $input2 -Oz -o $output2
## This only keeps SNPs but not sites that are both indels and SNPs.
echo "only keeps SNPs: no indels or others"
bcftools view -i 'TYPE="snp"' $output2 -Oz -o $dirtemp'temp_snp.'$input2_name
## Only keep biallelic SNPs
echo "Only keep biallelic SNPs"
# Based on manual: https://samtools.github.io/bcftools/bcftools.html
# -M, --max-alleles INT
# print sites with at most INT alleles listed in REF and ALT columns. Use -m2 -M2 -v snps to only view biallelic SNPs
bcftools view -m2 -M2 -v snps $dirtemp'temp_snp.'$input2_name -Oz -o $dirtemp'temp_bisnp.'$input2_name

# 1.3 Only keep masked sites using bedtools and maskfile
## load bedtools
echo "Step 1.3 Only keep masked sites using bedtools and maskfile"
echo "Load bedtools"
module load bedtools
## grep chr.maskfile to speed up
chrmask=$chr.$(basename "$maskfile")
###!!!!! use grep $chr will grep more than needed. 
###For example, chr1, chr11, chr12 will all be included if grep chr1. But this mistake is okay in this step. It just slows down the process.
###grep $chr $maskfile>$chrmask
grep "^$chr[[:space:]]" $maskfile>$chrmask
echo "Get mask file for "$chr" :"$chrmask
## 
input3=$dirtemp'temp_bisnp.'$input2_name
gzip -d $input3
input3="${input3%.*}"
bedtools intersect -a $input3 -b $chrmask -header | gzip > $output3
gzip $input3
echo "Get vcf file pass maskfile :"$output3

# 1.4 Check if there are duplicated sites in the mask.biallelic.vcf.gz file
echo "Step 1.4 Check if there are duplicated sites in the mask.biallelic.vcf.gz file: "$output3
## remove .vcf.gz extension from $output3 to name the simplified position txt file.
output3_name=$(basename "$output3")
output3_namenoext="${output3_name%.*}"
output3_namenoext="${output3_namenoext%.*}"
simppos=$dirtemp'temp_pos.'$output3_namenoext'.txt'
posdup=$dirtemp'temp_dupsites.'$output3_namenoext'.txt'
bcftools query -f '%CHROM\ %POS\n' $output3 >$simppos
sort $simppos | uniq -d > $posdup
if [ "$(cat $posdup| wc -l)" = 0 ]
then
echo "No duplicate sites in "$dirout$output3_name
rm $simppos $posdup
else
echo "Warning: duplicated sites in "$dirout$output3_name
echo "Please find duplicated sites in $posdup"
rm $simppos
fi

# 1.5 Record the number of sites in different steps
## If there is no duplicates. Record the number and file names.
### It's better to count the number of sites instead of number of lines by removing lines start with "#"
### The lines start with "#" for vcf.gz file can be countted by  zgrep -c '^#' $file
Linputvcf=$lines_inputvcf
Lsamples=$lines_output1
f3=$output2
L3="$(zcat $output2| wc -l)"
f4='temp_snp.'$input2_name
L4="$(zcat $dirtemp$f4| wc -l)"
f5='temp_bisnp.'$input2_name
L5="$(zcat $dirtemp$f5| wc -l)"
LmaskbiSNPs="$(zcat $output3| wc -l)"
cisite_counts="$inputvcf\t$Linputvcf\t$(basename "$output1")\t$Lsamples\t$f2\t$L2\t$f3\t$L3\t$f4\t$L4\t$f5\t$L5\t$output3\t$LmaskbiSNPs"
if [ "$(cat $dirtemp'temp_dupsites.'$output3_name| wc -l)" = 0 ]
then
echo -e "$chr\t$cisite_counts\t'no duplicated sites'" >>$output5
else
echo -e $chr\t$csites\t"check duplicated sites in "temp_dupsites.$output3_name >>$output5
fi













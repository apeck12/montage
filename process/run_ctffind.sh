# Script for running CTFFIND4 to estimate defocus, which takes as command line input:
# [input_file] [output_file] [pixel_size] [min_defocus] [max_defocus] [search_step]
# Note: below it's assumed that ctffind version 4.1.13 is available as a module.

# common microscope parameters
voltage=300.0
Cs=2.70
amp_contrast=0.07
spectrum_length=512
low_res=50.0

module load ctffind/4.1.13

ctffind <<EOF
$1
$2
$3
${voltage}
${Cs}
${amp_contrast}
${spectrum_length}
${low_res}
$7
$4
$5
$6
no
yes
yes
100
no
no
EOF

substring=$(echo $2 | cut -d'.' -f 1)
ctffind_plot_results.sh ${substring}_avrot.txt

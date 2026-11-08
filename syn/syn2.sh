#!/bin/bash
# usage: syn2.sh name top period_ps files...   (timing-driven, ORFS abc_speed script)
name=$1; top=$2; per=$3; shift 3
L=../lib
cat > $name.ys <<EOT
read_verilog -sv $@
hierarchy -check -top $top
synth -top $top -flatten
opt -purge
dfflibmap -liberty $L/SEQ_TT.lib
abc -script abc_speed.script -D $per -liberty $L/SIMPLE_TT.lib -liberty $L/INVBUF_TT.lib -dont_use *x1p*_ASAP7*
opt_clean -purge
hilomap -hicell TIEHIx1_ASAP7_75t_R H -locell TIELOx1_ASAP7_75t_R L
setundef -zero
splitnets -ports
opt_clean -purge
stat -liberty $L/SIMPLE_TT.lib -liberty $L/INVBUF_TT.lib -liberty $L/SEQ_TT.lib
write_verilog -noattr -noexpr -nohex -nodec $name.v
EOT
yowasp-yosys -q -l $name.log $name.ys

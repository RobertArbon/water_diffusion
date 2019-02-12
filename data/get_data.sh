#!/bin/bash

for j in {3..10}
do
scp ra15808@bluecrystalp3.bris.ac.uk:/projects/dynamics_of_glassy_aerosols/$j/"$j"_mw_disp_full.csv .
done

#/bin/bash


for i in 1 3 4 6 7 8 9 10
do
  papermill -p traj_num $i timescales.ipynb timescales-"$i".ipynb
done

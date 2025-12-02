set term png
set output 'execution_times.png'
set title 'Execution Times'
set xlabel 'Datasets'
set ylabel 'Execution Time'
set style data linespoints
set key top left

plot for [i=0:2] 'graph_data.dat' index i using 0:2 with linespoints title word(options, i+1)

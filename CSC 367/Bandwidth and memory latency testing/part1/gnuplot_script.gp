datafile1 = "collected_data_for_L1.dat"
datafile2 = "collected_data_for_L2.dat"
datafile3 = "collected_data_for_L3.dat"

set terminal png
set output "L1 cache size.png"
set key on
set ylabel "Cache latency (nanosecond)"
set xlabel "Estimated cache size (KB)"

plot datafile1 using 1:2 with lines linecolor "black" title "result", \
     "" using 1:2:1 with labels offset 1,1 notitle

set terminal png
set output "L2 cache size.png"
set key on
set ylabel "Cache latency (nanosecond)"
set xlabel "Estimated cache size (KB)"

plot datafile2 using 1:2 with lines linecolor "black" title "result", \
     "" using 1:2:1 with labels offset 1,1 notitle

set terminal png
set output "L3 cache size.png"
set key on
set ylabel "Cache latency (nanosecond)"
set xlabel "Estimated cache size (KB)"

plot datafile3 using 1:2 with lines linecolor "black" title "result", \
     "" using 1:2:1 with labels offset 1,1 notitle

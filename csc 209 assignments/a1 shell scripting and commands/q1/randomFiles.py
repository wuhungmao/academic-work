#!/usr/bin/python3
import sys, random, os

n = 8000
# disperse n files as 10000 files into directory00, ..., directory09

seed = int(sys.argv[1])
random.seed(seed) # for predictable results

alphabet = "abcdefghijklmnopqrstuvwxyz"
alpha_space = alphabet+"    "

# We create n file contents (these are the n images)
contents = []
for _ in range(n):
    file_len = random.randrange(500,1500)
    c = "".join([ alpha_space[random.randrange(len(alpha_space))] for _ in range(file_len) ])
    contents.append(c)

# We create n file names
file_names = []
for _ in range(n):
        file_name="".join([ alphabet[random.randrange(len(alphabet))] for _ in range(14) ] )
        file_names.append(file_name)

# Disperse them into directories
for i_dir in range(10):
    directory_name = "directory{:02d}".format(i_dir)
    os.mkdir(directory_name)

    for i_file in range(1000): # about 1000 files/directory
        which_filename = random.randrange(len(file_names))
        file_path_name = directory_name+"/"+file_names[which_filename]

        f = open(file_path_name,"w")
        which_contents = random.randrange(len(contents))
        f.write(contents[which_contents])
        f.close()


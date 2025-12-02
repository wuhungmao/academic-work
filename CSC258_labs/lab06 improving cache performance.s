# NAME:
# Q1:
# Configuration | Hit Rate  |   Hits   | Misses | Writebacks |
#       1       |  48.67%   |   128    |   135  |     31     |   
#       2       |  54.37%   |   143    |   120  |     10     |                  
#       3       |  46.01%   |   121    |   142  |     32     |           
#       4       |  22.43%   |    59    |   204  |     57     |   
#       5       |  50.19%   |   132    |   131  |     20     |   
#
#
# Q3:
# 2^n Lines: n=0
# 2^n Ways: n=2
# 2^n Blocks: n=3
# Hit Rate: 94.68%
# 
#
#

.data
newline:    .string      "\n"
delimiter:  .string      ", "
M:            .word 4
K:            .word 3
N:            .word 5
C:            .zero 80 # M * N * 4 bytes
.align 12 # M x K
A:            .word 4, 4, 3, 0, 1, 1, 3, 2, 4, 1, 1, 2
.align 15 # K x N
B:            .word 3, 2, 1, 0, 3, 5, 5, 4, 2, 5, 4, 5, 3, 0, 2
.text
main:
la s0, A
la s1, B
la s2, C
lw s3, M
lw s4, K
lw s5, N

li t0, 0            # M iteration index, m
li t1, 0            # K iteration index, k
li t2, 0            # N iteration index, n

# TODO: Q2
M_loop_head:
    # outer most loop start
    beq t0, s3, M_loop_end
    li t1, 0    # N index, n=0
    
K_loop_head:
    # inner most loop start
    beq t1, s4, K_loop_end
    li t2, 0    # K index, k=0
    
N_loop_head:
    # middle loop start
    beq t2, s5, N_loop_end
    
    # matrix multiplication...
    # m * N + n -> t3
    mul t3, t0, s5
    add t3, t3, t2
    slli t3, t3, 2 # index to byte offset
    add t3, s2, t3 # Location of C[m*N + n]
    lw a0, 0(t3) # Value of ^
    # m * K + k -> t4
    mul t4, t0, s4
    add t4, t4, t1
    slli t4, t4, 2
    add t4, s0, t4 # Location of A[m*K + k]
    lw a1, 0(t4) # Value of ^
    # k * N + n -> t5
    mul t5, t1, s5
    add t5, t5, t2
    slli t5, t5, 2
    add t5, s1, t5 # Location of B[k*N + n]
    lw a2, 0(t5) # Value of ^
    # A[m*K + k] * B[k*N + n] -> t6
    mul t6, a1, a2
    # C[m*N + n] += A[m*K + k] * B[k*N + n], store result in a0
    add a0, a0, t6 
    sw a0, 0(t3) # store the result into C[m*N + n], in memory
    
    addi t2, t2, 1 # k++
    j N_loop_head

N_loop_end:
    # middle loop end
    addi t1, t1, 1 # m++
    j K_loop_head
       
K_loop_end:
    # inner most loop end
    addi t0, t0, 1 # n++
    j M_loop_head
    

M_loop_end:
    # outer most loop end
    # total iterations is M * N, store in s6
    mul s6, s3, s5
    # index counter is stored in t0
    li t0, 0 # m=0
    
print_row_head:
    beq t0, s3, print_row_end
    li t1, 0 # n=0
    
print_col_head:
    beq t1, s5, print_col_end
    # compute offset, store in t2
    mul t2, t0, s5
    add t2, t2, t1
    slli t2, t2, 2
    add t2, s2, t2 # compute address of base + offset
    jal print_value
    jal print_delimiter
    addi t1, t1, 1 # n++
    j print_col_head
    
print_col_end:
    addi t0, t0, 1 # m++
    jal print_new_line
    j print_row_head
    
print_row_end:
    j exit
    
print_value:
    # print the vale of C[i]
    lw a0, 0(t2) # print the value of C[i]
    li a7, 1
    ecall
    jr x1
    
print_delimiter:
    la a0, delimiter
    li a7, 4
    ecall
    jr x1
    
print_new_line:
    la a0, newline
    li a7, 4
    ecall
    jr x1

exit:
    # exit program gracefully
    li a7, 10
    ecall
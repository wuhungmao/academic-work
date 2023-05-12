.data
# TODO: What are the following 5 lines doing?
promptN: .string "TOO MANY TIMES"
N: .word 5
promptinput: .string "input"
array1: .word 5, 8, 3, 4, 7, 2
newline: .string "\n"

.globl main
.text

main:
LOOPINIT:
li t1, 6
li t2, 1
la t0, array1
li t4, 1

WHILE:
beq t2, t1, DONE
lw t3, 0(t0)
mul t4, t4, t3
addi t0, t0, 4
addi t2, t2, 1
j WHILE

ELSE2:
li a7, 4
la a0, promptN
ecall
j DONE
DONE:
li a7, 10
ecall

readInt:
    addi sp, sp, -12
    li a0, 0
    mv a1, sp
    li a2, 12
    li a7, 63
    ecall
    li a1, 1
    add a2, sp, a0
    addi a2, a2, -2
    mv a0, zero
parse:
    blt a2, sp, parseEnd
    lb a7, 0(a2)
    addi a7, a7, -48
    li a3, 9
    bltu a3, a7, error
    mul a7, a7, a1
    add a0, a0, a7
    li a3, 10
    mul a1, a1, a3
    addi a2, a2, -1
    j parse
parseEnd:
    addi sp, sp, 12
    ret

error:
    li a7, 93
    li a0, 1
    ecall

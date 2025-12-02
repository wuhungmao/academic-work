.data
# TODO: What are the following 5 lines doing?
promptodd: .string "THIS IS ODD"
prompteven: .string "THIS IS EVEN"
promptA: .string "Enter an int A\n"
promptB: .string "Enter an int B\n"
promptN: .string "TOO MANY TIMES"
N: .word 5
promptinput: .string "input"
resultAdd: .string "A + B + C = "
resultSub: .string "A - B = "
newline: .string "\n"

.globl main
.text

main:
LOOPINIT:
lw t1, N
WHILE:
li a7, 4
la a0, promptinput
ecall
call readInt
mv t0, a0
andi t0, t0, 1
li t2, 1
li t3, 0
beq t3, t0, DONE
sub t1, t1, t2
beq t1, t3, ELSE2
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

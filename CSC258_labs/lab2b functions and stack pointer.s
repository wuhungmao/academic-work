.data 
promptinput: .string "input"
promptoutout: .string "Enter a number\n"
promptoutput2: .string "The result is "

.globl main
.text

main:
    li a7, 4
    la a0, promptinput
    ecall
    
    li a7, 1
    call readInt
    
    li t5, 0
    jal mystery
    
    mv a0, t5
    li a7, 1
    ecall
    
    li a7, 10
    ecall
    
mystery:
    addi sp, sp, -6
    sw ra, 0(sp)
    sb a0, 4(sp)
    li t3, 2
    mul t1, a0, t3
    sb t1, 5(sp)
    li t2, 0
    beq a0, t2, return0

returncompute:

    li t4, 1
    sub a0, a0, t4
    jal mystery
    lw ra, 0(sp)
    lb a0, 4(sp)
    lb t1, 5(sp)
    addi t5, t5, -1
    add t5, t5, t1
    addi sp, sp, 6
    jr ra
    
return0:
    lw ra, 0(sp)
    addi sp, sp, 6
    jr ra
    
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

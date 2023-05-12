# NAME:
# Q1: 
# Q3: 
.data
foo:    .word 15
.text
# TODO Q2: 

#Q1: The high/low signal I highlighted is write enable in register file, it is signalled when addi x11 x0 2 is executed while it is not when sw x10 0 x0 is executed. The reason is because sw does not change value in any register but addi does change value in x11. 
#Q2: It is a comparing instruction evaluated to be true like bltz t1, label where t1 = -1. 
#Q3: if write enable is low, then it means no value change in registers. If write enable is high in data memory, then it means instruction stores value into memory, like sw instruction. The instruction is load instruction with an offset, like lw t0, (1)a0.
main:
    li t1, -1
    bltz t1, begin
    sw a0 0(zero)	  
    li a1, 2
    li t0, 10
begin: 	  
    add a0, a0, a1
    and a2, a0, a1
end:
    lw a0, foo
    li a7, 10
    ecall
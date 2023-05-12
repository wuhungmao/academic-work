# NAME:Wu Hung Mao
# Q1:
# 1. 
# a.Instruction fetch
# b.Fetch instruction from instruction memory
# c.pc, instruction memory
# 2.
# a.Instruction decode
# b.retrieve data from register file and extend immediate
# c.Register file and immediate generator
# 3.
# a.Execution
# b.do arithmetic operation in ALU. The data is from register file
# c.ALU, Mux, Add ALU
# 4.
# a.Memory access
# b.retrieve data from memory or write data to memory
# c.Data memory
# 5.
# a.Write back
# b.Write data retrieved from data memory to register 
# c.Mux, Register file
#
# Q2: 
# c. 5-stage processor cannot run without nop instruction
# d. 
# Num_Cycles 5-stage w/o forwarding or hazard detection: 25
# Num_Cycles 5-stage: 16
# Speedup: 1.5625
#
# Q3:
# Single-Cycle Processor:
#    a. 1050
#    b. 1050
#    c. 1
# 5-stage Processor w/o Forwarding Unit
#    a. 300
#    b. 1800
#    c. 1.75
# 5-stage Processor (Pipelined)
#    a. 300
#    b. 1500
#    c. 1.94

.text     
main:
    # TODO Q2a.: Write the value of the array next to each corresponding sw instruction.
    # TODO Q2b.: Add noops to make this work as expected.
    auipc s0, 0x10000 # load the memory address at which we store the array
    li x3, 7
    li x4, 15
    #nop
    sw x3, 0(s0)  #0
    addi x3, x3, 2
    #nop
    #nop
    sw x3, 4(s0) #7
    addi x3, x3, 2
    #nop
    #nop
    sw x3, 8(s0) #9
    addi x3, x3, 2
    #nop
    #nop
    sw x3, 12(s0) #9
    addi x3, x3, 2
    #nop
    #nop
    sw x3, 16(s0) #11
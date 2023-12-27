.data
sequence:  .byte 0,0,0,0,0,0,0,0,0,0
count:     .word 4
directionmessage1: .string "up\n"
directionmessage2: .string "left\n"
directionmessage3: .string "down\n"
directionmessage4: .string "right\n"
promptmessage1: .string "please enter your answer\n"
errormessage: .string "wrong you lose!!!"
successmessage: .string "success you win!!! do you want to play one more round 1 for yes, 0 for no\n"
.globl main
.text


    # TODO: Before we deal with the LEDs, we need to generate a random
    # sequence of numbers that we will use to indicate the button/LED
    # to light up. For example, we can have 0 for UP, 1 for DOWN, 2 for
    # LEFT, and 3 for RIGHT. Store the sequence in memory. We provided 
    # a declaration above that you can use if you want.
    # HINT: Use the rand function provided to generate each number

#questions: when the user run the program, is he playing it in fast execution mode?
# cheat sheet 1=down=red , 2=left=green, 0=up=yellow, 3=right=blue

    firstround:
        li s0, 1000
        li s2, 4
        la gp, 0x000003b8
        j assign_random_number_initialization
        
    anotherround:
        la gp, 0x000003b8
        li t0, 2
        li t3, 1
        addi s2, s2, 1
        blt s0, t3, assign_random_number_initialization
        div s0, s0, t0
    
    # stores four random generated number into the sequence
    
    assign_random_number_initialization:
        addi t1, gp, 0
        addi t4, s2, 0
        li t2, 1
        li t3, 0
        
    assign_random_number:
        beq t4, t3, Init
        li a0, 4
        jal rand
        sb a0, (0)t1
        addi t1, t1, 1
        sub t4, t4, t2
        j assign_random_number
    # TODO: Now read the sequence and replay it on the LEDs. You will
    # need to use the delay function to ensure that the LEDs light up 
    # slowly. In general, for each number in the sequence you should:
    # 1. Figure out the corresponding LED location and colour
    # 2. Light up the appropriate LED (with the colour)
    # 2. Wait for a short delay (e.g. 500 ms)
    # 3. Turn off the LED (i.e. set it to black)
    # 4. Wait for a short delay (e.g. 1000 ms) before repeating
    
    Init:
        addi t6, gp, 0
        lb t5, 0(t6)
        li t1, 0
        li t2, 1
        li t3, 2
        li t4, 3
        #a4 is counter
        addi a4, s2, 0
        
    while1:
        beq t5, t1, yellow
        beq t5, t2, red
        beq t5, t3, green
        beq t5, t4, blue
        addi, t6, t6, 1
        lb t5, 0(t6)
        sub a4, a4, t2
        beq a4, t1, while1Done
        j while1
        
    yellow:
        la a3, 0x0
        la a0, 0xffff00
        li a1, 0
        li a2, 0
        jal setLED
        li t1, 0
    
        addi a0, s0, 0
        jal delay
        li t1, 0
        li t2, 1
        
        la a3, 0x0
        lb a0, (0)a3
        li a1, 0
        li a2, 0
        jal setLED
        li t1, 0
        li a0, 1000
        jal delay
        li t1, 0
        li t2, 1
        li t5, 5
        j while1
    
    red:
        la a3, 0x0
        la a0, 0x99ff99
        li a1, 1
        li a2, 1
        jal setLED
        li t1, 0
    
        addi a0, s0, 0
        jal delay
        li t1, 0
        li t2, 1
    
        la a3, 0x0
        lb a0, (0)a3
        li a1, 1
        li a2, 1
        jal setLED
        li t1, 0
        li t5, 5
        li a0, 1000
        jal delay
        li t1, 0
        li t2, 1
        j while1
    
    green:
        la a3, 0x0
        la a0, 0x00cc00
        li a1, 1
        li a2, 0
        jal setLED
        li t1, 0
    
        addi a0, s0, 0
        jal delay
        li t1, 0
        li t2, 1
    
        la a3, 0x0
        lb a0, (0)a3
        li a1, 1
        li a2, 0
        jal setLED
        li a0, 1000
        jal delay
        li t1, 0
        li t2, 1
        li t5, 5
        j while1
    
    blue:
        la a3, 0x0
        la a0, 0x3333ff
        li a1, 0
        li a2, 1
        jal setLED
        li t1, 0
    
        addi a0, s0, 0
        jal delay
        li t1, 0
        li t2, 1
    
        la a3, 0x0
        lb a0, (0)a3
        li a1, 0
        li a2, 1
        jal setLED
        li a0, 1000
        jal delay
        li t1, 0
        li t2, 1
        li t5, 5
        j while1
        
    while1Done:
    # TODO: Read through the sequence again and check for user input
    # using pollDpad. For each number in the sequence, check the d-pad
    # input and compare it against the sequence. If the input does not
    # match, display some indication of error on the LEDs and exit. 
    # Otherwise, keep checking the rest of the sequence and display 
    # some indication of success once you reach the end.
    
    # cheat sheet 1=down=red , 2=left=green, 0=up=yellow, 3=right=blue
    
    initwhile2:
        li a7, 4
        la a0, promptmessage1
        ecall 
        
        li t2, 0
        li t3, 1
        addi t4, s2, 0
        addi t6, gp, 0
        lb t5, 0(t6)
        
    while2:
        jal pollDpad
        li t2, 0
        li t3, 1
        mv t0, a0
        jal printdirection
        bne t0, t5, errorcode
        addi t6, t6, 1
        lb t5, 0(t6)
        sub t4, t4, t3
        beq t4, t2, success
        j while2
        
    errorcode:
        li a7, 4
        la a0, errormessage
        ecall
        j exit
        
    success:
        li a7, 4
        la a0, successmessage
        ecall
        call readInt
        li t0, 0
        li t1, 1
        beq t0, a0, exit
        beq t1, a0, anotherround
        j exit
        
    printdirection:
        li s3, 2
        li s4, 3
        # 1=down=red , 2=left=green, 0=up=yellow, 3=right=blue
        beq t0, t2, up
        beq t0, t3, down
        beq t0, s3, left
        beq t0, s4, right
        
    up:
        li a7, 4
        la a0, directionmessage1
        ecall
        jr ra
    left:
        li a7, 4
        la a0, directionmessage2
        ecall
        jr ra
    down:
        li a7, 4
        la a0, directionmessage3
        ecall
        jr ra
    right:
        li a7, 4
        la a0, directionmessage4
        ecall
        jr ra


    # TODO: Ask if the user wishes to play again and either loop back to
    # start a new round or terminate, based on their input.
    
    exit:
        li a7, 10
        ecall
    
    
# --- HELPER FUNCTIONS ---
# Feel free to use (or modify) them however you see fit
     
# Takes in the number of milliseconds to wait (in a0) before returning
delay:
    mv t0, a0
    li a7, 30
    ecall
    mv t1, a0
delayLoop:
    ecall
    sub t2, a0, t1
    bgez t2, delayIfEnd
    addi t2, t2, -1
delayIfEnd:
    bltu t2, t0, delayLoop
    jr ra

# Takes in a number in a0, and returns a (sort of) random number from 0 to
# this number (exclusive)
rand:
    mv t0, a0
    li a7, 30
    ecall
    remu a0, a0, t0
    jr ra
    
# Takes in an RGB color in a0, an x-coordinate in a1, and a y-coordinate
# in a2. Then it sets the led at (x, y) to the given color.
setLED:
    li t1, LED_MATRIX_0_WIDTH
    mul t0, a2, t1
    add t0, t0, a1
    li t1, 4
    mul t0, t0, t1
    li t1, LED_MATRIX_0_BASE
    add t0, t1, t0
    sw a0, (0)t0
    jr ra
    
# Polls the d-pad input until a button is pressed, then returns a number
# representing the button that was pressed in a0.
# The possible return values are:
# 0: UP
# 1: DOWN
# 2: LEFT
# 3: RIGHT

# cheat sheet 1=down=red , 2=left=green, 0=up=yellow, 3=right=blue

pollDpad:
    mv a0, zero
    li t1, 4
pollLoop:
    bge a0, t1, pollLoopEnd
    li t2, D_PAD_0_BASE
    slli t3, a0, 2
    add t2, t2, t3 
    lw t3, (0)t2
    bnez t3, pollRelease
    addi a0, a0, 1
    j pollLoop
pollLoopEnd:
    j pollDpad
pollRelease:
    lw t3, (0)t2
    bnez t3, pollRelease
pollExit:
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

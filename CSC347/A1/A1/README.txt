Include any other files that are necessary to make your exploits or fixes work and run properly. (supply as few files as possible to make marking easy)
Include any README.txt or REPORT.txt files which will explain what your files are supposed to be doing.

Student name 1: mumark
Student name 2: Wu Hung Mao
Student name 3: leedogyu

a1/race/
  explanation.txt (#3 in race.c)
  modifyRoot.txt (#5 in race.c)
  rootShell.txt (#6 in race.c)
  race.c (#7 in race.c)
  race_old.c # this is the old, vulnerable version of the program

a1/vulnerable/
  account_old.c # the old, vulnerable version of the program
  account.c # this is the fixed version of the program
  README.txt # installation instructions
  setup.bash # installation script
  accounts
  log
  passwords
  REPORT.txt # this is the file containing a report of vulnerabilities,
  # exploits and impact, see the sample file

  # any other files you need to make your new program work

a1/vulnerable/exploits/
  # a collection of exploits for the account system
  # we should be able to run these against our original installation
  e1
  e2
  e3
  ...

a1/palindrome/
  tcpclientA.pl
  tcpclientB.pl
  tcpclientB.pl
  palindrome.c # the fixed version of the palindrome program
  palindrome_old.c # the original version of the palindrome program
  palindrome.xinetd
  REPORT.txt
  questionC
  # any other files needed

a1/mystery/
  instructions.txt

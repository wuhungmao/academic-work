#!/bin/bash
gcc -o palindrome palindrome.c
echo "done compiling"
mv palindrome /root/a1
echo "done mv palindrome to /root/a1"
cd /root/a1
echo "done cd to /root/a1"
chmod 755 palindrome
echo "done chmod palindrome"
/etc/rc.d/init.d/xinetd restart
echo "done restart the service"

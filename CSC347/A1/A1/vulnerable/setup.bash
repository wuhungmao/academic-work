#!/bin/bash
# make sure you do this as root, and all files and directories
# are owned by root, or just use sudo

cd "$(dirname "$0")"
chmod 755 .

rm -f account *.o core core.*
rm -fr accounts
rm -f log passwords

make clean
make account
chmod 755 account
chmod +s account

mkdir accounts
touch log
touch passwords
chmod 700 accounts
chmod 600 log passwords


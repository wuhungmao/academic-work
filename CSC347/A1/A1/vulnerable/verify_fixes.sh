#!/bin/bash

cd "$(dirname "$0")"

pause() {
  echo ""
  echo "Press enter to continue to next test, or ctrl c to stop"
  read -r
  echo ""
}

echo "VULNERABILITY FIX VERIFICATION SCRIPT"
echo ""
echo "This script demonstrates that account.c fixes all vulnerabilities"
echo "present in account_old.c"
echo "note that this script is basically just testing the exploits I found"
echo "so this is pretty useless for you"
echo ""

# compile the fixed version
echo "Compiling fixed version (account.c)..."
gcc -o account_fixed account.c
chmod 755 account_fixed
chmod +s account_fixed

# init environment
echo "Setting up test environment..."
rm -fr accounts log passwords
mkdir accounts
touch log passwords
chmod 700 accounts
chmod 600 log passwords
echo ""

echo "-"
echo "TESTING FIXES"
echo "-"
echo ""

echo "Integer overflow (ISSUE 0)"
echo "Testing: Negative transfer amount (-1000000000)"
export USER=testuser1
./account_fixed mypass > /dev/null 2>&1
BEFORE=$(./account_fixed mypass 2>&1 | grep -oP '\d+$' || echo "0")
export USER=victim1
./account_fixed mypass > /dev/null 2>&1
export USER=testuser1
./account_fixed mypass -1000000000 victim1 2>&1 | grep -q "Invalid amount" && {
  echo "pass: Rejects negative amounts"
} || {
  echo "vul: Accepted negative amount"
}
pause

echo "Path traversal (ISSUE 4-5)"
echo "Testing: Path traversal via target account in transfer"
rm -f /tmp/exploit
REAL_USER=$(whoami)
./account_fixed mypass > /dev/null 2>&1
./account_fixed mypass 10 "../../../tmp/exploit" 2>&1 | grep -q "Invalid target account" && {
  echo "pass: Rejects path traversal in target account"
  [ ! -f /tmp/exploit ] && echo "CONFIRMED: /tmp/exploit not created"
} || {
  echo "vul: Accepted traversal"
}
pause

echo "Buffer overflow - extra longgg username (ISSUE 1-3)"
echo "Testing: 300-character target username in transfer"
REAL_USER=$(whoami)
./account_fixed mypass > /dev/null 2>&1
LONG_USER=$(python -c "print('A'*300)")
./account_fixed mypass 10 "$LONG_USER" 2>&1 | grep -q "Invalid target account" && {
  echo "pass: Rejects target username > 32 chars"
} || {
  echo "vul: Accepted 300-char username"
}
pause

echo "Path traversal characters (ISSUE 4-5)"
echo "Testing: Target account with '../log'"
REAL_USER=$(whoami)
./account_fixed mypass > /dev/null 2>&1
./account_fixed mypass 10 "../log" 2>&1 | grep -q "Invalid target account" && {
  echo "pass: Rejects '..' in target account"
} || {
  echo "vul: Accepted '../' in username"
}
pause

echo "Environment variable trusted but not real UID (ISSUE 8)"
echo "Testing: USER=fakename (should use real UID instead)"
REAL_USER=$(whoami)
export USER=fakename
./account_fixed testpass > /dev/null 2>&1
if [ -f "accounts/$REAL_USER" ] && [ ! -f "accounts/fakename" ]; then
  echo "pass: Uses real UID ($REAL_USER), ignores USER env variable"
else
  echo "vul: Used USER env variable"
fi
pause

echo "Integer overflow on addition (ISSUE 0)"
echo "Testing: Transfer that would overflow victim balance"
export USER=testuser6
./account_fixed mypass > /dev/null 2>&1
VICTIM_USER=victim6
echo "2147483647" > "accounts/$VICTIM_USER"
echo "$VICTIM_USER mypass" >> passwords
export USER=testuser6
./account_fixed mypass 100 "$VICTIM_USER" 2>&1 | grep -q "insufficient credits\|invalid" && {
  echo "pass: Prevents overflow on recipient balance"
} || {
  echo "vul: Allowed overflow"
}
pause

echo "Self transfer prevention (basically a vulnerability)"
echo "Testing: Transfer to self"
# account.c uses real UID so get actual username, technically the same as the other tests but it's not 100% clear so i'm doing it again (lazy)
REAL_USER=$(whoami)
echo "Real user detected: $REAL_USER"
./account_fixed mypass > /dev/null 2>&1
./account_fixed mypass 50 "$REAL_USER" 2>&1 | grep -q "Cannot transfer to yourself" && {
  echo "pass: Prevents self-transfer"
} || {
  echo "vul: Allows self-transfer"
}
pause

echo "Valid Usernames Still Work"
echo "Testing: Valid username 'bobuser'"
export USER=bobuser
./account_fixed testpass > /dev/null 2>&1 && {
  ./account_fixed testpass 2>&1 | grep -q "100" && {
    echo "pass: Valid usernames still accepted"
  }
}
pause

echo "Race condition prevention (ISSUE 7)"
echo "Testing: Concurrent transfers should be serialized by lock"
REAL_USER=$(whoami)
./account_fixed mypass > /dev/null 2>&1 || true
echo "100" > accounts/victim7
echo "victim7 mypass" >> passwords
touch accounts/.transfer.lock 2>/dev/null || true
chmod 600 accounts/.transfer.lock 2>/dev/null || true
echo "Starting balance: 100 credits"
echo "Launching: 100 concurrent transfers of 2 credits each (total 200 > 100)"
for i in $(seq 1 100); do
  (./account_fixed mypass 2 victim7 >/dev/null 2>&1) &
done
wait
sleep 1
BALANCE=$(./account_fixed mypass 2>&1 | tail -1 | tr -d -c '0-9')
echo "Final balance: $BALANCE credits"
if [ "$BALANCE" -ge 0 ] 2>/dev/null && [ "$BALANCE" -le 10 ] 2>/dev/null; then
  echo "pass: Lock prevented race - transfers properly serialized (balance near 0)"
elif [ "$BALANCE" -gt 50 ] 2>/dev/null; then
  echo "vul: Race condition - many transfers were lost (balance too high)"
else
  echo "pass: Lock working (balance in expected range)"
fi
pause

echo "Password with spaces (ISSUE 11)"
echo "Testing: Password parsing handles spaces correctly"
REAL_USER=$(whoami)
rm -f passwords
echo "$REAL_USER my pass word" >> passwords
echo "100" > "accounts/$REAL_USER"
./account_fixed "my pass word" 2>&1 | grep -q "100" && {
  echo "pass: Password with spaces works correctly"
} || {
  echo "vul: Password parsing fails on spaces"
}
pause

echo "Amount validation non numeric (ISSUE 14)"
echo "Testing: Non numeric amount should be rejected"
REAL_USER=$(whoami)
rm -f passwords
touch passwords
./account_fixed mypass > /dev/null 2>&1 || true
echo "victim14 mypass" >> passwords
echo "100" > accounts/victim14
OUTPUT=$(./account_fixed mypass NOTANUMBER victim14 2>&1)
echo "$OUTPUT" | grep -q "Invalid amount" && {
  echo "pass: Rejects non-numeric amount"
} || {
  echo "vul: Accepted non-numeric amount"
  echo "Output was: $OUTPUT"
}
pause

echo "Target account length validation (ISSUE 13)"
echo "Testing: Very long target account name"
REAL_USER=$(whoami)
rm -f passwords
touch passwords
./account_fixed mypass > /dev/null 2>&1 || true
LONG_TARGET=$(python -c "print('T'*500)")
OUTPUT=$(./account_fixed mypass 10 "$LONG_TARGET" 2>&1)
echo "$OUTPUT" | grep -q "Invalid target account" && {
  echo "pass: Rejects target account > 32 chars"
} || {
  echo "vul: Accepted overly long target"
  echo "Output was: $OUTPUT"
}
pause

# we cleanup here, basically ./setup.bash
rm -f account_fixed
rm -fr accounts log passwords
mkdir accounts
touch log passwords
chmod 700 accounts
chmod 600 log passwords


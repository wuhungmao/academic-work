#!/bin/bash

set -e
cd "$(dirname "$0")"

pause() {
  echo ""
  echo "Press enter to continue to next comparison, or ctrl c to exit"
  read -r
  echo ""
}

echo "Compiling both versions..."
gcc -o account_vulnerable account_old.c 2>/dev/null
gcc -o account_fixed account.c 2>/dev/null
chmod 755 account_vulnerable account_fixed
chmod +s account_vulnerable account_fixed
echo ""

run_test() {
  local test_name="$1"
  local test_cmd="$2"
  
  echo "TEST: $test_name"
  
  rm -fr accounts log passwords
  mkdir accounts
  touch log passwords
  chmod 700 accounts
  chmod 600 log passwords
  
  echo "[vul ver - account_old.c]:"
  ln -sf account_vulnerable account
  eval "$test_cmd" 2>&1 | head -20
  echo ""
  
  rm -fr accounts log passwords
  mkdir accounts
  touch log passwords
  chmod 700 accounts
  chmod 600 log passwords
  
  echo "[fixed ver - account.c]:"
  ln -sf account_fixed account
  eval "$test_cmd" 2>&1 | head -20
  echo ""
  rm -f account
  
  pause
}

run_test "Integer overflow (negative transfer)" '
REAL_USER=$(whoami)
./account mypass >/dev/null 2>&1
echo "victim mypass" >> passwords
echo "100" > accounts/victim
echo ""
echo "Attempting: transfer -1000000000 to victim"
./account mypass -1000000000 victim 2>&1 | head -10
'

run_test "Path traversal (create account with path)" '
rm -f /tmp/pwned
export USER="../../../../../tmp/pwned"
echo ""
echo "Attempting: Create account with USER=\"../../../../../tmp/pwned\""
./account testpass 2>&1 | head -5
echo ""
if [ -f /tmp/pwned ]; then
  echo "path traversal worked! File created at /tmp/pwned"
  ls -la /tmp/pwned
  rm -f /tmp/pwned
else
  echo "path traversal blocked, file not created"
fi
'

run_test "Buffer overflow from long target username" '
REAL_USER=$(whoami)
./account mypass >/dev/null 2>&1
LONG=$(python -c "print(\"A\"*300)")
echo ""
echo "Attempting: transfer to 300 char username"
./account mypass 10 "$LONG" 2>&1 | head -5
'

run_test "Self transfer exploit" '
REAL_USER=$(whoami)
./account mypass >/dev/null 2>&1
echo ""
echo "Attempting: transfer 50 to self ($REAL_USER)"
./account mypass 50 "$REAL_USER" 2>&1 | head -10
'

run_test "Race condition (concurrent transfers)" '
REAL_USER=$(whoami)
echo "Testing race condition with extreme concurrency..."
rm -f accounts/* passwords
touch passwords
./account mypass >/dev/null 2>&1 || true
echo "100" > accounts/victim
echo "victim mypass" >> passwords
echo "Starting balance: 100 credits"
echo "Launching 200 concurrent transfers of 1 credit each..."
echo "expecting 0 but both will likely be close to there anyways me thinks"
for i in $(seq 1 200); do
  ./account mypass 1 victim >/dev/null 2>&1 &
done
wait
sleep 2
BALANCE=$(./account mypass 2>&1 | tail -1 | tr -d -c "0-9")
echo "Final balance: $BALANCE credits"
if [ "$BALANCE" -gt 70 ] 2>/dev/null; then
  echo "rc detected!!?!?!"
fi
'

run_test "Password with spaces" '
REAL_USER=$(whoami)
rm -f passwords
touch passwords
echo "$REAL_USER my secure pass" >> passwords
echo "100" > "accounts/$REAL_USER"
echo ""
echo "Attempting: Authenticate with password containing spaces"
./account "my secure pass" 2>&1 | head -10
'

run_test "Non numeric amount validation" '
REAL_USER=$(whoami)
./account mypass >/dev/null 2>&1 || true
echo "victim mypass" >> passwords
echo "100" > accounts/victim
echo ""
echo "Attempting: Transfer with amount NOTANUMBER"
./account mypass NOTANUMBER victim 2>&1 | head -10
'

run_test "Recipient balance overflow" '
REAL_USER=$(whoami)
./account mypass >/dev/null 2>&1 || true
echo "1000" > "accounts/$REAL_USER"
echo "victim mypass" >> passwords
echo "2147483647" > accounts/victim
echo ""
echo "Attacker has: 1000 credits"
echo "Victim has: 2147483647 (INT_MAX)"
echo "Attempting: Transfer 100 to account with INT_MAX balance"
./account mypass 100 victim 2>&1 | head -10
'

rm -f account_vulnerable account_fixed account
rm -fr accounts log passwords
mkdir accounts
touch log passwords
chmod 700 accounts
chmod 600 log passwords
Hashing algorithm = md5crypt
Password: 
1. butter
2. is_4
3. popcorn
Commands: 
john --format=md5crypt --wordlist=/usr/share/wordlists/rockyou.txt --rules a2q4.txt
// found butter and popcorn
john --format=md5crypt --mask='?1?1?1?1' --1='?d?l?s' a2q4.txt
// found is_4
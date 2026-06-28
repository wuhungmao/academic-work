Wordlists: 
    weakpass_4.txt: https://weakpass.com/download/2012/weakpass_4.txt.7z

1. $1$xyz$/gkzk7Xbmpk0MDz1wxr6W0
Algorithm: 
    md5crypt
Password: 
    password1234!   
Commands: 
    // a2q5h1.txt contains the first hash. 
    hashcat -m 1800 -w 3 -O -a 0 a2q5h1.txt weakpass_4.txt 

2. $6$xyz$CSrEEoXkwhdxwcLAf5lMU0D2/VkIAyEJm.KZHV8tD6g0PnpfuRrRq5D/.OYKWJVkvzpWoaIGSm2mRYEHRKiQe0
Algorithm: 
    sha512crypt
Password: 
Commands: 
    // a2q5h2.txt contains the second hash. 
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/best66.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/combinator.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/leetspeak.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/oscommerce.rule	
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/specific.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/stacking58.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/T0XlC_3_rule.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/T0XlC-insert_space_and_special_0_F.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/T0XlC-insert_top_100_passwords_1_G.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/toggles1.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/toggles2.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/toggles2.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/toggles4.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt rockyou.txt -r rules/top10_2025.rule
    ./hashcat.bin -m 1800 -w 3 -O -a 0 a2q5h2.txt weakpass_4.txt

3. $5$xyz$zKaWsgEvhG8BNTJvTxx6zq4GcoGVpUsyR4vLgGciRaA
Algorithm: 
    sha256crypt
Password: 
Commands: 
    // a2q5h3.txt contains the third hash. 
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/best66.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/combinator.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/leetspeak.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/oscommerce.rule	
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/specific.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/stacking58.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/T0XlC_3_rule.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/T0XlC-insert_space_and_special_0_F.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/T0XlC-insert_top_100_passwords_1_G.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/toggles1.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/toggles2.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/toggles2.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/toggles4.rule 
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt rockyou.txt -r rules/top10_2025.rule
    ./hashcat.bin -m 7400 -w 3 -O -a 0 a2q5h3.txt weakpass_4.txt

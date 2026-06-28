#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>

/**

1) as root, compile race.c (use the makefile), make race
    cd /questions/race
    make

2) Verify that hacker can not view or edit /etc/passwd2

3) Explain how do you check whether the setuid bit is checked on a file?

4) As hacker, run /questions/race/race to understand how it works and what it does

5) As hacker, explain how hacker exploits /questions/race/race
   so that they modify the root only readable and writable file /etc/passwd2
	 Where is this coming from? Why could there be a race condition?

6) As hacker, explain how hacker exploits /questions/race/race
   so that they end up with a root shell

   Be careful when doing this you may lock yourself out of the system!!
   I have made a backup of /etc/passwd in /etc/passwd.bak
   make sure you have a root shell when playing with the passwd
   file.

7) Fix the code below so that it eliminates the race condition and
   allows race to safely write to /tmp/permitted, no matter which user
   owns the file.
	 Depending on which fix you choose to implement, provide a written explanation of	why this fixes the issue (max 200 words).

References:
http://www.cis.syr.edu/~wedu/seed/Labs_12.04/Software/Race_Condition/Race_Condition.pdf
http://www.csl.mtu.edu/cs3451/www/notes/ch6%20-%20Adding%20new%20users.pdf

**/

int main(int argc, char ** argv) {
	char * fn = "/tmp/permitted";
	char buffer[128];
	FILE *fp;
	struct stat file_stat_before;
	struct stat file_stat_after;

	// Store original file's inode number
	stat(fn, &file_stat_before);

	// No matter which user owns the file => Check marking scheme
	// It in itself eliminates race conditions by dropping privileges based on each user's id (uid)
	seteuid(getuid());

	// access doesn't fail because hacker is owner of /tmp/permitted
	if(!access(fn, W_OK)){
		scanf("%100s", buffer);

		// Check current fn's inode number
		stat(fn, &file_stat_after);
		if (file_stat_before.st_ino != file_stat_after.st_ino) {
			printf("Different file \n");
			// return 1;
		}

		fp = fopen(fn, "w");
		// fp NULL if getuid was not root who owns the target file
		// this is where seteuid() matters
		if (fp == NULL) {
				perror("No permission\n");
				return 1;
		}

		// No symlink was created
		fwrite(buffer, sizeof(char), strlen(buffer), fp);
		fwrite("\n", sizeof(char), 1, fp);
		fclose(fp);
	} else {
		printf("No permission \n");
	}
}

/** Explanation for 7)
To eliminate the race condition, stop the hacker from using the symlink to overwrite the passwd file to actually login as root. This can be achieved by setting the euid to the hacker's uid using seteuid(getuid()). This reduces hacker privileges, so it cannot not run on root privileges. access() passes since hacker owns /tmp/permitted; however, it can't overwrite the /etc/passwd since the euid isn't root's anymore. Thus, fopen() doesn't open the target file and returns NULL handled by the if condition with perror and an early return. The symlink exists, so the hacker can read /etc/passwd, but cannot physically edit /etc/passwd to login as root. If no symlink was created, users can safely write to the original file.

My secondary approach was to use inode numbers, since each file has a unique inode number. [This approach stops all symlinks (I commented out the early return); Andi said to keep it] The code compares the inode numbers (st_ino) of file_stat_before and file_stat_after (stored before and after access()). If they match, the fopen() succeeds and the user can write onto it. If they don't match, it indicates there was a symlink (Need to include <sts/stat.h>)
**/

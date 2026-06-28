#include <stdio.h>
#include <stdlib.h>
// my stuff
#include <string.h>
#include <unistd.h>
#include <pwd.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/file.h>
// --

/**

You are doing a code review on the following credit trading system...

As root, on the RH7.2 Virtual Machine

	cd /vulnerable
	# replace the account.c with this one!!!!
	# chmod +x setup.bash
	./setup.bash

NOTE: Make sure the permssions are correct after you recompile account!

Make sure permissions are as follows: ls -al 

total 52
drwxr-xr-x    3 root     root         4096 Oct  4 22:02 .
drwxr-xr-x   20 root     root         4096 Oct  4 20:35 ..
-rwsr-sr-x    1 root     root        17459 Oct  4 21:58 account
-rw-------    1 root     root         4317 Oct  4 22:02 account.c
drwx------    2 root     root         4096 Oct  4 21:59 accounts
-rw-------    1 root     root          105 Oct  4 22:00 log
-rw-------    1 root     root           27 Oct  4 21:59 passwords
-rwx------    1 root     root          176 Oct  4 21:30 setup

Each user can now execute

/vulnerable/account myPassword   # to create my account with 100 credits, initialized with myPassword
/vulnerable/account myPassword 20 otherUser # give 20 credits to otherUser

1) Identify any bufferoverrun, integer overflow, canonical naming, priviledge escalation, denial of service etc.
   issues in this code. Submit a copy of the code annotated with the issues. To make it easier to
   find your annotations write ISSUE: before each issue you identify.
2) Demonstrate that the above vulnerabilities can be exploited and list the potential outcomes.
3) Fix the code so that the vulnerabilities are eliminated, or describe how the vulnerability/exploit
   should be addressed.


*/

// my stuff helpersss
#define ACCOUNT_DIR "accounts"
#define PASSWORDS_PATH "passwords"
#define LOG_PATH "log"

static int isValidUsername(const char *user){
	size_t len;
	size_t i;
	if(user==NULL) return 0;
	/* 1..32 chars, [A-Za-z0-9_] */
	len = strlen(user);
	if(len == 0 || len > 32) return 0;
	for(i = 0; i < len; i++){
		char c = user[i];
		if(!(isalnum((unsigned char)c) || c=='_')) return 0;
	}
	return 1;
}

static int buildAccountPath(const char *user, char *out, size_t outSize){
	int n;
	if(!isValidUsername(user)) return -1;
	n = snprintf(out, outSize, ACCOUNT_DIR "/%s", user);
	if(n < 0 || (size_t)n >= outSize) return -1;
	return 0;
}

static int acquireTransferLock(void){
	int fd;
	fd = open(ACCOUNT_DIR "/.transfer.lock", O_CREAT|O_RDWR, 0600);
	if(fd < 0) return -1;
	if(flock(fd, LOCK_EX) < 0){
		close(fd);
		return -1;
	}
	return fd;
}

static void releaseTransferLock(int fd){
	if(fd >= 0){
		flock(fd, LOCK_UN);
		close(fd);
	}
}
// -- below is also my stuff but mixed into original code

// The user is not in the system, so add them and their password
static int addUser(const char *user, const char *password){
	FILE * file;
	if(!isValidUsername(user)) return -1;
	if(password==NULL) return -1;
	file=fopen(PASSWORDS_PATH,"a");
	if(!file) return -1;
    // would use hashing in prod tbh
	fprintf(file, "%s %s\n", user, password);
	fclose(file);
	return 0;
}

static int getAccount(const char * user){
	FILE * file;
	char fileName[PATH_MAX];
	int amount=0;
	if(buildAccountPath(user, fileName, sizeof(fileName))!=0) return -1;
	file=fopen(fileName, "r");
	if(file){
		if(fscanf(file, "%d", &amount)!=1){
			amount = -1;
		}
		fclose(file);
	} else {
		return -1; // to signify that an account does not exist
	}
	return amount;
}

static int setAccount(const char * user, int amount){
	char fileName[PATH_MAX];
	int fd;
	FILE *file;
	if(buildAccountPath(user, fileName, sizeof(fileName))!=0) return -1;
	/* Safely create/overwrite with 0600 permissions */
	fd = open(fileName, O_WRONLY|O_CREAT|O_TRUNC, 0600);
	if(fd < 0) return -1;
	file = fdopen(fd, "w");
	if(!file){
		close(fd);
		return -1;
	}
	fprintf(file, "%d", amount);
	fclose(file);
	// perms
	chmod(fileName, 0600);
	return 0;
}

static int logTransaction(const char * transaction){
	FILE * file;
	if(!transaction) return -1;
	file=fopen(LOG_PATH,"a");
	if(!file) return -1;
	fprintf(file, "%s\n", transaction);
	fclose(file);
	return 0;
}

static int authenticate(const char *user, const char *password){
	FILE * file;
	char line[256];
	char u[100], p[100];
	char *space;
	if(!isValidUsername(user)) return 0;
	if(!password) return 0;
	file=fopen(PASSWORDS_PATH,"r");
	if(!file){
		// no passwords? treat not found
		return 2;
	}
	while(fgets(line, sizeof(line), file)){
		space = strchr(line, ' ');
		if(!space) continue;
		*space = '\0';
		strncpy(u, line, sizeof(u)-1);
		u[sizeof(u)-1] = '\0';
		strncpy(p, space+1, sizeof(p)-1);
		p[sizeof(p)-1] = '\0';
		p[strcspn(p, "\n")] = '\0';
		
		if(strncmp(user,u,100)==0){
			if(strncmp(password, p, 100)==0){
				fclose(file);
				return 1;
			}else{
				fclose(file);
				return 0;
			}
		}
	}
	fclose(file);
	return 2;
}

static int report(const char * user){
	int c;
	FILE * file;
	char fileName[PATH_MAX];
	if(buildAccountPath(user, fileName, sizeof(fileName))!=0) return -1;
	file=fopen(fileName, "r");
	if(!file){
		printf("account %s does not exist\n", user);
		return -1;
	}
	while((c=fgetc(file))!=EOF){
		putchar(c);
	}
	fclose(file);
	printf("\n");
	return 0;
}

int main(int argc, char *argv[]){

	char user[100];
	char password[100];
	char transaction[2048];
	int auth;
	int i;
	size_t used;
	struct passwd *pw;
	long parsed;
	char *endptr;
	int amount;
	int fromAmount, toAmount;
	int lockfd;

	if(argc!=2 && argc!=4){
		printf("account password (to setup/report on your account)\n");
		printf("account password amount targetAccount (to transfer)\n");
		return 0;
	}

	// user from real uid, not enviro
	pw = getpwuid(getuid());
	if(!pw || !pw->pw_name){
		printf("Unable to determine user.\n");
		return 0;
	}
	strncpy(user, pw->pw_name, sizeof(user)-1);
	user[sizeof(user)-1] = '\0';
	if(!isValidUsername(user)){
		printf("Invalid username.\n");
		return 0;
	}

	/* for auditing purposes */
	used = 0;
	transaction[0]='\0';
	used += (size_t)snprintf(transaction+used, sizeof(transaction)-used, "%s: ", user);
	for(i=1;i<argc && used < sizeof(transaction);i++){
		used += (size_t)snprintf(transaction+used, sizeof(transaction)-used, "%s ", argv[i]);
	}

	strncpy(password,argv[1],sizeof(password)-1);
	password[sizeof(password)-1]='\0';
    // password length validation
	if(strlen(password)==0){
		printf("Empty password not allowed.\n");
		return 0;
	}

	auth=authenticate(user, password);

	if(argc==2){ 
		if(auth==2){
			addUser(user, password);
			setAccount(user,100);
			printf("Your account has:\n");
			report(user);
		} else if(auth==1){
			printf("Your account has:\n");
			report(user);
		} else {
			printf("You have not been authenticated\n");
		}
	} else if(argc==4){ // perform a transfer to another account
		if(auth==1){
			errno = 0;
			endptr = NULL;
			parsed = strtol(argv[2], &endptr, 10);
			if(errno!=0 || endptr==argv[2] || *endptr!='\0' || parsed < 0 || parsed > INT_MAX){
				printf("Invalid amount.\n");
				logTransaction(transaction);
				return 0;
			}
			amount = (int)parsed;

			/* validate target account name and ensure not self */
			if(!isValidUsername(argv[3])){
				printf("Invalid target account.\n");
				logTransaction(transaction);
				return 0;
			}
		if(strncmp(user, argv[3], sizeof(user))==0){
			printf("Cannot transfer to yourself.\n");
			logTransaction(transaction);
			return 0;
		}

		lockfd = acquireTransferLock();
		if(lockfd < 0){
			printf("Unable to acquire transfer lock.\n");
			logTransaction(transaction);
			return 0;
		}

		fromAmount=getAccount(user);
		toAmount=getAccount(argv[3]);
		if(toAmount==-1){
			printf("account %s does not exist\n",argv[3]);
			releaseTransferLock(lockfd);
		} else if(fromAmount>=amount && toAmount <= INT_MAX - amount){
			printf("Your account had:\n");
			report(user);

			fromAmount=fromAmount-amount;
			toAmount=toAmount+amount;
			setAccount(user,fromAmount);
			setAccount(argv[3],toAmount);

			printf("Your account now has:\n");
			report(user);
			releaseTransferLock(lockfd);
		} else {
			printf("You do not have sufficient credits or amount invalid.\n");
			releaseTransferLock(lockfd);
		}
		} else { 
			printf("You have not been authenticated\n");
		} 
	}

	/* in any case, log the attempt */
	logTransaction(transaction);

	return 0;
}

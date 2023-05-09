#include<stdio.h>
#include<sys/types.h>
#include<unistd.h>
#include<stdlib.h>

#define TRUE 1
#define FALSE 0

// return whether n has a divisor in i,j
int has_divisor(unsigned long n, unsigned long i, unsigned long j){
	while(i<=j){
		if(n%i==0)return TRUE;
		i=i+1;
	}
	return FALSE;
}

// determine if argv[1] is prime
int main(int argc, char ** argv){
	unsigned long n=atoll(argv[1]);

	int is_prime=FALSE;
	
	if(n==2 || n==3){
		is_prime=TRUE;
	} else if(n>3 && !has_divisor(n, 2,n-1)){
		is_prime=TRUE;
	}
	if(is_prime){
		printf("%lu is prime\n", n);
	} else {
		printf("%lu is not prime\n", n);

	}
	exit(is_prime); 
}


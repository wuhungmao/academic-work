#include <stdio.h>
#include <string.h>

char *secret = "you got me!!";

// return 1 if we should continue, 0 if we should stop
int processOneLine()
{
	char s[1024];
	char *s1, *s2;
	int isPalindrome = 1; // it is a palindrome until we find out otherwise
	int count;
	count = 0;
	s1 = s;
	while (count < 1024)
	{
		*s1 = getchar();
		if (*s1 == '\n')
			*s1 = '\0';
		if (*s1 == '\0')
			break;
		s1++;
		count++;
	}
	if (count == 1024)
	{
		*s1 = '\0';
	}
	s2 = s;
	s1--;
	count = 0;
	while (s2 < s1 && count < 1024)
	{
		if (*s1 != *s2)
		{
			isPalindrome = 0;
			break;
		}
		s1--;
		s2++;
		count++;
	}
	printf("%s", s);
	if (isPalindrome)
	{
		printf(" is a palindrome\n");
	}
	else
	{
		printf(" is not a palindrome\n");
	}
	fflush(stdout);
	if (strncmp(s, "quit", 4) == 0)
		return 0;
	return 1;
}

int main(int argc, char **argv)
{
	printf("Palindrome server, 'quit' to quit:\n");
	fflush(stdout);
	while (1)
	{
		if (!processOneLine())
			break;
	}
	return (0);
}

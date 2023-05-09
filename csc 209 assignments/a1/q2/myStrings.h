/**
 * header file for myStrings.c
 * Include in both myStrings.c and in any src that links with myStrings.o
 */


/**
 * copy characters of src to dst (including '\0')
 * We assume tht the caller has sufficient space!
 */
void myStrCpy(char dst[], char src[]);

/**
 * returns the length of string s
 * this is the number of non-null characters in s
 */
int myStrLen(char s[]);

/**
 * returns whether *s is the empty string
 */
int myisStrEmpty(char s[]);

/**
 * returns whether *s1 is equal to *s2
 */
int myisStrEQ(char s1[], char s2[]);


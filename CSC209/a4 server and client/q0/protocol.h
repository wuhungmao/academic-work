#define MAX_USER_LEN 64
#define MAX_CHAT_MESSAGE_LEN (1024)
#define MAX_MESSAGE_LEN (7+1+MAX_USER_LEN+1+MAX_USER_LEN+1+MAX_CHAT_MESSAGE_LEN+1)
// Note: We assume that the \n or the \0 is at worst, the last character in buffer[MAX_MESSAGE_LEN]

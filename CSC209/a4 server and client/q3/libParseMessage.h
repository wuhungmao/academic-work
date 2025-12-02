/**
 * Split buffer into at most 4 null terminated parts, 
 * part[0]...part[3] by placing '\0' inside buffer and 
 * returning pointers to the pieces. 
 *
 * params
 *      char *buffer: a null terminated string
 *      char *part[]: array of 4 pointers to char
 * return
 *      the number of pieces parsed
 *      0 if this is not a valid message
 */

int parseMessage(char *buffer, char *part[]);

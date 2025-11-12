#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    char rem[50], a[100], s[100], c, msg[50], gen[30];
    int i, genlen, t, j, flag = 0, k, n;

    printf("Enter the generator polynomial (e.g., 11021 for CRC-CCITT is 10001000000100001):\n");
    scanf("%s", gen);

    printf("Generator polynomial (CRC-CCITT): %s\n", gen);
    genlen = strlen(gen);
    k = genlen - 1;

    printf("Enter the message bits (e.g., 1011001):\n");
    scanf("%s", msg);
    n = strlen(msg);

    // Copy message and append k zeros
    for (i = 0; i < n; i++)
        a[i] = msg[i];
    for (i = 0; i < k; i++)
        a[n + i] = '0';
    a[n + k] = '\0';

    printf("\nMessage polynomial appended with %d zeros:\n", k);
    puts(a);

    // Division to find remainder
    for (i = 0; i < n; i++) {
        if (a[i] == '1') {
            t = i;
            for (j = 0; j < genlen; j++, t++) {
                a[t] = (a[t] == gen[j]) ? '0' : '1';
            }
        }
    }

    // Extract remainder
    for (i = 0; i < k; i++)
        rem[i] = a[n + i];
    rem[k] = '\0';

    printf("The CRC (remainder) is:\n");
    puts(rem);

    // Append CRC to message
    for (i = 0; i < n; i++)
        a[i] = msg[i];
    for (i = 0; i < k; i++)
        a[n + i] = rem[i];
    a[n + k] = '\0';

    printf("\nTransmitted message (data + CRC):\n");
    puts(a);

    // Receiver side check
    printf("\nEnter the received message bits:\n");
    scanf("%s", s);
    n = strlen(s);

    // Division on received message
    for (i = 0; i < n - k; i++) {
        if (s[i] == '1') {
            t = i;
            for (j = 0; j < genlen; j++, t++) {
                s[t] = (s[t] == gen[j]) ? '0' : '1';
            }
        }
    }

    // Extract remainder after checking
    for (i = 0; i < k; i++)
        rem[i] = s[n - k + i];
    rem[k] = '\0';

    flag = 0;
    for (i = 0; i < k; i++) {
        if (rem[i] == '1')
            flag = 1;
    }

    if (flag == 0)
        printf("\n✅ Received polynomial is error-free.\n");
    else
        printf("\n❌ Error detected in received polynomial.\n");

    return 0;
}

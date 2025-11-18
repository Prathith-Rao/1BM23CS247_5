/* client_udp.c
 * Usage: ./client_udp <server-ip> <port>
 * Example: ./client_udp 127.0.0.1 7777
 *
 * Simple UDP client:
 *  - sends filename to server using sendto()
 *  - receives datagrams from server and prints to stdout until "END_OF_FILE" received
 *  - prints error if receives "FILE_NOT_FOUND"
 *
 * Note: Without reliability mechanisms, some packets may be lost. This is a lab example.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>

#define MAX_FILENAME 512
#define DATA_BUFSZ 2048      /* socket buffer for recvfrom (bigger than server chunk) */
#define CONTROL_EOF "END_OF_FILE"
#define CONTROL_NOTFOUND "FILE_NOT_FOUND"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <server-ip> <port>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *server_ip = argv[1];
    int port = atoi(argv[2]);
    int sfd = -1;
    struct sockaddr_in servaddr;
    socklen_t serv_len = sizeof(servaddr);

    sfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sfd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    if (inet_pton(AF_INET, server_ip, &servaddr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid server IP: %s\n", server_ip);
        close(sfd);
        exit(EXIT_FAILURE);
    }

    char filename[MAX_FILENAME];
    printf("Enter the file with complete path: ");
    if (!fgets(filename, sizeof(filename), stdin)) {
        fprintf(stderr, "No filename entered\n");
        close(sfd);
        exit(EXIT_FAILURE);
    }
    size_t len = strlen(filename);
    if (len > 0 && (filename[len-1] == '\n' || filename[len-1] == '\r')) {
        filename[len-1] = '\0';
        len--;
    }
    if (len == 0) {
        fprintf(stderr, "Empty filename\n");
        close(sfd);
        exit(EXIT_FAILURE);
    }

    /* send filename to server */
    ssize_t sent = sendto(sfd, filename, len, 0, (struct sockaddr *)&servaddr, serv_len);
    if (sent < 0) { perror("sendto"); close(sfd); exit(EXIT_FAILURE); }

    /* receive datagrams until EOF marker */
    char buf[DATA_BUFSZ];
    ssize_t n;
    printf("\n--- Received from server ---\n");
    while (1) {
        n = recvfrom(sfd, buf, sizeof(buf)-1, 0, (struct sockaddr *)&servaddr, &serv_len);
        if (n < 0) {
            perror("recvfrom");
            break;
        }
        buf[n] = '\0';

        /* check control messages */
        if (n == (ssize_t)strlen(CONTROL_EOF) && strcmp(buf, CONTROL_EOF) == 0) {
            /* finished */
            break;
        }
        if (n == (ssize_t)strlen(CONTROL_NOTFOUND) && strcmp(buf, CONTROL_NOTFOUND) == 0) {
            fprintf(stderr, "Server: file not found\n");
            break;
        }

        /* write received bytes to stdout; note buf is NUL-terminated above for safety */
        ssize_t out = 0;
        while (out < n) {
            ssize_t w = write(STDOUT_FILENO, buf + out, n - out);
            if (w < 0) { perror("write stdout"); close(sfd); exit(EXIT_FAILURE); }
            out += w;
        }
    }
    printf("\n--- End ---\n");

    close(sfd);
    return 0;
}

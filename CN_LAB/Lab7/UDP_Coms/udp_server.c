/* server_udp.c
 * Usage: ./server_udp <port>
 * Example: ./server_udp 7777
 *
 * Simple UDP file server:
 *  - listens for a filename from a client
 *  - sends file contents back to the client's address using sendto()
 *  - sends "FILE_NOT_FOUND" if file can't be opened
 *  - sends "END_OF_FILE" after finishing the transfer
 *
 * Note: This is a simple lab example. UDP is unreliable; packets may be lost.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>

#define MAX_FILENAME 512
#define DATA_BUFSZ 1024        /* payload per datagram */
#define CONTROL_EOF "END_OF_FILE"
#define CONTROL_NOTFOUND "FILE_NOT_FOUND"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <port>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int port = atoi(argv[1]);
    int sfd = -1;
    struct sockaddr_in servaddr, cliaddr;
    socklen_t cli_len = sizeof(cliaddr);

    /* create UDP socket */
    sfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sfd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(port);

    if (bind(sfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("bind");
        close(sfd);
        exit(EXIT_FAILURE);
    }

    printf("UDP server listening on port %d ...\n", port);

    /* receive filename from client */
    char filename[MAX_FILENAME];
    ssize_t n = recvfrom(sfd, filename, sizeof(filename)-1, 0,
                         (struct sockaddr *)&cliaddr, &cli_len);
    if (n < 0) { perror("recvfrom"); close(sfd); exit(EXIT_FAILURE); }
    filename[n] = '\0';
    /* trim newline if sent */
    if (n > 0 && (filename[n-1] == '\n' || filename[n-1] == '\r')) filename[n-1] = '\0';

    printf("Request from %s:%d -> filename: '%s'\n",
           inet_ntoa(cliaddr.sin_addr), ntohs(cliaddr.sin_port), filename);

    /* open file */
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        ssize_t sent = sendto(sfd, CONTROL_NOTFOUND, strlen(CONTROL_NOTFOUND), 0,
                              (struct sockaddr *)&cliaddr, cli_len);
        if (sent < 0) perror("sendto notfound");
        close(sfd);
        exit(EXIT_FAILURE);
    }

    /* send file contents in chunks */
    char buf[DATA_BUFSZ];
    size_t read_bytes;
    while ((read_bytes = fread(buf, 1, sizeof(buf), fp)) > 0) {
        ssize_t sent = sendto(sfd, buf, read_bytes, 0,
                              (struct sockaddr *)&cliaddr, cli_len);
        if (sent < 0) {
            perror("sendto");
            fclose(fp);
            close(sfd);
            exit(EXIT_FAILURE);
        }
        /* In a real protocol you may wait for ACKs; we don't here. */
    }

    /* send EOF marker */
    if (sendto(sfd, CONTROL_EOF, strlen(CONTROL_EOF), 0,
               (struct sockaddr *)&cliaddr, cli_len) < 0) {
        perror("sendto EOF");
    }

    printf("Transfer complete (or at least finished sending). Closing.\n");

    fclose(fp);
    close(sfd);
    return 0;
}

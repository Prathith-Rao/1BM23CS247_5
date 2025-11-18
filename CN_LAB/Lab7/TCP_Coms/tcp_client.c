/* client.c
 * Usage: ./client <server-ip> <port>
 * Example: ./client 127.0.0.1 7777
 *
 * Simple client:
 *  - connects to server
 *  - reads filename from stdin
 *  - sends filename to server
 *  - reads data from server and writes it to stdout
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define REQ_BUFSZ 256
#define IO_BUFSZ 4096

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Err: usage:\n\t./client <server-ip> <port>\n\tex: ./client 127.0.0.1 7777\n");
        exit(EXIT_FAILURE);
    }

    const char *server_ip = argv[1];
    int portno = atoi(argv[2]);
    int sockfd = -1;
    struct sockaddr_in serv;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    memset(&serv, 0, sizeof(serv));
    serv.sin_family = AF_INET;
    serv.sin_port = htons(portno);
    if (inet_pton(AF_INET, server_ip, &serv.sin_addr) <= 0) {
        fprintf(stderr, "Invalid server IP: %s\n", server_ip);
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (connect(sockfd, (struct sockaddr *)&serv, sizeof(serv)) < 0) {
        perror("connect");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    char filename[REQ_BUFSZ];
    printf("Enter the file with complete path: ");
    if (!fgets(filename, sizeof(filename), stdin)) {
        fprintf(stderr, "No filename entered\n");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    /* remove newline */
    size_t len = strlen(filename);
    if (len > 0 && (filename[len-1] == '\n' || filename[len-1] == '\r')) {
        filename[len-1] = '\0';
        len--;
    }
    if (len == 0) {
        fprintf(stderr, "Empty filename\n");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    /* send filename */
    ssize_t sent = write(sockfd, filename, len);
    if (sent < 0) {
        perror("write filename");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    /* read server response and print to stdout */
    char buf[IO_BUFSZ];
    ssize_t r;
    printf("\n--- File contents / server message ---\n");
    while ((r = read(sockfd, buf, sizeof(buf))) > 0) {
        ssize_t out = 0;
        while (out < r) {
            ssize_t w = write(STDOUT_FILENO, buf + out, r - out);
            if (w < 0) { perror("write stdout"); close(sockfd); exit(EXIT_FAILURE); }
            out += w;
        }
    }
    if (r < 0) perror("read from socket");

    printf("\n--- end ---\n");
    close(sockfd);
    return 0;
}

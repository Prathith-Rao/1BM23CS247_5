/* server.c
 * Usage: ./server <port>
 * Example: ./server 7777
 *
 * Simple single-connection file server:
 *  - listens on given port
 *  - accepts one client
 *  - reads filename (up to 255 bytes)
 *  - opens file and streams contents to client (using read/write)
 *  - sends "FILE_NOT_FOUND\n" if file can't be opened
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

#define BACKLOG 5
#define REQ_BUFSZ 256
#define IO_BUFSZ 4096

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "error: no port no\nusage:\n\t./server <port>\n");
        exit(EXIT_FAILURE);
    }

    int portno = atoi(argv[1]);
    int sockfd = -1, newsockfd = -1;
    struct sockaddr_in serv, cli;
    socklen_t len = sizeof(cli);

    /* create socket */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    /* bind */
    memset(&serv, 0, sizeof(serv));
    serv.sin_family = AF_INET;
    serv.sin_addr.s_addr = INADDR_ANY;
    serv.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *)&serv, sizeof(serv)) < 0) {
        perror("bind");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (listen(sockfd, BACKLOG) < 0) {
        perror("listen");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("server: waiting for connection on port %d ...\n", portno);
    newsockfd = accept(sockfd, (struct sockaddr *)&cli, &len);
    if (newsockfd < 0) {
        perror("accept");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("connection from %s:%d\n", inet_ntoa(cli.sin_addr), ntohs(cli.sin_port));

    /* read filename */
    char filename[REQ_BUFSZ];
    ssize_t n = read(newsockfd, filename, sizeof(filename)-1);
    if (n < 0) {
        perror("read filename");
        close(newsockfd);
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    filename[n] = '\0';
    /* trim newline if any */
    if (n > 0 && (filename[n-1] == '\n' || filename[n-1] == '\r')) filename[n-1] = '\0';

    printf("server received filename: '%s'\n", filename);

    /* open file */
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        char *msg = "FILE_NOT_FOUND\n";
        if (write(newsockfd, msg, strlen(msg)) < 0) perror("write error");
        fprintf(stderr, "server: cannot open file '%s': %s\n", filename, strerror(errno));
        close(newsockfd);
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    /* stream file to socket */
    char buf[IO_BUFSZ];
    ssize_t r;
    while ((r = read(fd, buf, sizeof(buf))) > 0) {
        ssize_t written = 0;
        while (written < r) {
            ssize_t w = write(newsockfd, buf + written, r - written);
            if (w < 0) {
                perror("write to socket");
                close(fd);
                close(newsockfd);
                close(sockfd);
                exit(EXIT_FAILURE);
            }
            written += w;
        }
    }
    if (r < 0) perror("read file");

    printf("transfer complete\n");

    close(fd);
    close(newsockfd);
    close(sockfd);
    return 0;
}

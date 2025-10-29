#include <stdio.h>
#include <string.h>

int min(int x, int y)
{
    if (x < y)
        return x;
    else
        return y;
}

int main()
{
    int drop = 0, mini, nsec, cap, count = 0, i, inp[25], process;

    printf("Enter the bucket size:\n");
    scanf("%d", &cap);

    printf("Enter the processing rate:\n");
    scanf("%d", &process);

    printf("Enter the number of seconds you want to simulate:\n");
    scanf("%d", &nsec);

    for (i = 0; i < nsec; i++)
    {
        printf("Enter the size of the packet entering at %d sec:\n", i + 1);
        scanf("%d", &inp[i]);
    }

    printf("\nSecond | Packet received | Packet sent | Packet left | Dropped\n");
    printf("-----------------------------------------------------------------\n");

    for (i = 0; i < nsec; i++)
    {
        count += inp[i];
        if (count > cap)
        {
            drop = count - cap;
            count = cap;
        }

        printf("%6d | %15d |", i + 1, inp[i]);

        mini = min(count, process);
        printf(" %11d |", mini);

        count -= mini;
        printf(" %11d | %7d\n", count, drop);

        drop = 0;
    }

    // process remaining packets after input ends
    while (count != 0)
    {
        if (count > cap)
        {
            drop = count - cap;
            count = cap;
        }

        printf("%6d | %15d |", i + 1, 0);

        mini = min(count, process);
        printf(" %11d |", mini);

        count -= mini;
        printf(" %11d | %7d\n", count, drop);

        drop = 0;
        i++;
    }

    return 0;
}

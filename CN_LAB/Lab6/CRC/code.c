#include <stdio.h>
#include <stdint.h>

#define POLY 0x1021   // CRC-CCITT polynomial (x^16 + x^12 + x^5 + 1)
#define INIT 0xFFFF   // Initial value for CRC

// Function to compute CRC-16-CCITT
uint16_t crc16_ccitt(const int data_bits[], int n) {
    uint16_t crc = INIT;

    for (int i = 0; i < n; i++) {
        int bit = data_bits[i] & 1;          // Current data bit
        int msb = (crc >> 15) & 1;           // MSB of CRC
        crc <<= 1;                           // Shift left

        if (bit ^ msb)
            crc ^= POLY;                     // XOR with polynomial if bits differ
    }
    return crc & 0xFFFF;                     // Return 16-bit CRC
}

int main() {
    int data_bits[1024];
    int n;

    printf("Enter number of bits in data: ");
    scanf("%d", &n);

    printf("Enter %d bits (0 or 1):\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &data_bits[i]);
        if (data_bits[i] != 0 && data_bits[i] != 1) {
            printf("Invalid input! Only 0 or 1 allowed.\n");
            return 1;
        }
    }

    // Compute CRC
    uint16_t crc = crc16_ccitt(data_bits, n);

    printf("\nComputed CRC-CCITT (16-bit): ");
    for (int i = 15; i >= 0; i--) {
        printf("%d", (crc >> i) & 1);
    }
    printf("\n");

    // Simulate received frame
    int recv_bits[1040];
    for (int i = 0; i < n; i++)
        recv_bits[i] = data_bits[i];

    // Append CRC bits to data
    for (int i = 0; i < 16; i++)
        recv_bits[n + i] = (crc >> (15 - i)) & 1;

    printf("\nTransmitted Frame (data + CRC): ");
    for (int i = 0; i < n + 16; i++)
        printf("%d", recv_bits[i]);
    printf("\n");

    // Receiver side check
    uint16_t check_crc = crc16_ccitt(recv_bits, n + 16);
    if (check_crc == 0)
        printf("\n✅ No error detected in received frame.\n");
    else
        printf("\n❌ Error detected! CRC remainder = 0x%04X\n", check_crc);

    return 0;
}

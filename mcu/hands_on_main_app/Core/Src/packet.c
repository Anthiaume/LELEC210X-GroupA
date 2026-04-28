/*
 * packet.c
 */

#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include <string.h>

const uint8_t AES_Key[16]  = {
                            0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00};


// void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {

//     // Buffer 16 bytes (4 * 32 bits)
//     uint32_t statew[4] = {0};
//     // state is a pointer to the start of the buffer
//     uint8_t *state = (uint8_t*) statew;
//     size_t i;

//     // Nombre de blocs (arrondi supérieur)
//     size_t n = (msg_len + 15) / 16;

//     // Cas message vide
//     if (msg_len == 0) {
//         for (int j = 0; j < 16; j++) {
//             tag[j] = 0;
//         }
//         return;
//     }

//     for (i = 0; i < n; i++) {

//         uint8_t block[16] = {0};

//         // Taille réelle du bloc courant
//         size_t remaining = msg_len - i * 16;
//         size_t block_len = (remaining >= 16) ? 16 : remaining;

//         // Copie + zero padding implicite
//         for (int j = 0; j < block_len; j++) {
//             block[j] = msg[i * 16 + j];
//         }

//         // XOR avec l'état précédent
//         for (int j = 0; j < 16; j++) {
//             state[j] ^= block[j];
//         }

//         // Chiffrement AES
//         AES128_encrypt(state, AES_Key);
//     }

//     // Copie du tag
//     for (int j = 0; j < 16; j++) {
//         tag[j] = state[j];
//     }
// }


/* Pack 16 bytes (big-endian per word) into 4 x 32-bit words */
static void pack_block_be(const uint8_t in[16], uint32_t w[4])
{
    w[0] = ((uint32_t)in[0] << 24) |
           ((uint32_t)in[1] << 16) |
           ((uint32_t)in[2] <<  8) |
           ((uint32_t)in[3]      );
    w[1] = ((uint32_t)in[4] << 24) |
           ((uint32_t)in[5] << 16) |
           ((uint32_t)in[6] <<  8) |
           ((uint32_t)in[7]      );
    w[2] = ((uint32_t)in[8] << 24) |
           ((uint32_t)in[9] << 16) |
           ((uint32_t)in[10] << 8) |
           ((uint32_t)in[11]     );
    w[3] = ((uint32_t)in[12] << 24) |
           ((uint32_t)in[13] << 16) |
           ((uint32_t)in[14] << 8) |
           ((uint32_t)in[15]     );
}

/* Unpack 4 x 32-bit words (big-endian per word) into 16 bytes */
static void unpack_block_be(const uint32_t w[4], uint8_t out[16])
{
    out[0]  = (uint8_t)(w[0] >> 24);
    out[1]  = (uint8_t)(w[0] >> 16);
    out[2]  = (uint8_t)(w[0] >>  8);
    out[3]  = (uint8_t)(w[0]      );
    out[4]  = (uint8_t)(w[1] >> 24);
    out[5]  = (uint8_t)(w[1] >> 16);
    out[6]  = (uint8_t)(w[1] >>  8);
    out[7]  = (uint8_t)(w[1]      );
    out[8]  = (uint8_t)(w[2] >> 24);
    out[9]  = (uint8_t)(w[2] >> 16);
    out[10] = (uint8_t)(w[2] >>  8);
    out[11] = (uint8_t)(w[2]      );
    out[12] = (uint8_t)(w[3] >> 24);
    out[13] = (uint8_t)(w[3] >> 16);
    out[14] = (uint8_t)(w[3] >>  8);
    out[15] = (uint8_t)(w[3]      );
}

/* Init AES peripheral for 128-bit CBC encryption, IV = 0, DATATYPE = 32-bit */
static void AES_HW_Init_CBC(const uint8_t *key16)
{
    /* 1. Enable AES clock */
    RCC->AHB2ENR |= RCC_AHB2ENR_AESEN;
    (void)RCC->AHB2ENR; // dummy read

    /* 2. Disable AES */
    AES->CR = 0;

    /* 3. Configure:
     * MODE = 00 (encryption)
     * CHMOD = 001 (CBC)
     * KEYSIZE = 0 (128-bit)
     * DATATYPE = 00 (no swap, we pack big-endian)
     */
    AES->CR =
        (0U << AES_CR_MODE_Pos)    |  // encryption
        (1U << AES_CR_CHMOD_Pos)   |  // CBC
        (0U << AES_CR_KEYSIZE_Pos) |  // 128-bit key
        (0U << AES_CR_DATATYPE_Pos);  // 32-bit, no swap

    /* 4. Load key into KEYR registers (big-endian per word) */
    uint32_t kw[4];
    pack_block_be(key16, kw);
    AES->KEYR0 = kw[0];
    AES->KEYR1 = kw[1];
    AES->KEYR2 = kw[2];
    AES->KEYR3 = kw[3];

    /* 5. IV = 0 for CBC-MAC */
    AES->IVR0 = 0;
    AES->IVR1 = 0;
    AES->IVR2 = 0;
    AES->IVR3 = 0;
}

/* CBC-MAC using STM32 AES hardware (128-bit key, zero IV, zero padding) */
void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len)
{
    /* Cas message vide : tag = 0 */
    if (msg_len == 0) {
        memset(tag, 0, 16);
        return;
    }

    AES_HW_Init_CBC(AES_Key);

    /* Enable AES */
    AES->CR |= AES_CR_EN;

    size_t n = (msg_len + 15U) / 16U;
    uint8_t  block[16];
    uint32_t w_in[4];
    uint32_t w_out[4];

    for (size_t i = 0; i < n; i++) {

        /* Préparer bloc avec padding zéro */
        memset(block, 0, 16);
        size_t remaining = msg_len - i * 16U;
        size_t block_len = (remaining >= 16U) ? 16U : remaining;

        for (size_t j = 0; j < block_len; j++) {
            block[j] = msg[i * 16U + j];
        }

        /* Pack en 4 mots 32 bits big-endian */
        pack_block_be(block, w_in);

        /* Écrire dans DINR (4 words) */
        AES->DINR = w_in[0];
        AES->DINR = w_in[1];
        AES->DINR = w_in[2];
        AES->DINR = w_in[3];

        /* Attendre fin de traitement (CCF) */
        while ((AES->SR & AES_SR_CCF) == 0U) {
            /* wait */
        }

        /* Lire résultat (4 words) */
        w_out[0] = AES->DOUTR;
        w_out[1] = AES->DOUTR;
        w_out[2] = AES->DOUTR;
        w_out[3] = AES->DOUTR;

        /* Clear CCF */
        AES->CR |= AES_CR_CCFC;
    }

    /* Désactiver AES */
    AES->CR &= ~AES_CR_EN;

    /* Dernier bloc chiffré = tag */
    uint8_t last_block[16];
    unpack_block_be(w_out, last_block);
    memcpy(tag, last_block, 16);
}



// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial)
{
	// printf("make_packet called with payload_len=%u, sender_id=%u, serial=%u\n", payload_len, sender_id, serial);
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;

    // Initially, the whole packet header is set to 0s
    memset(packet, 0, PACKET_HEADER_LENGTH);

    // So is the tag
    memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);

    // ✅ CORRECTION 1 : packet[0] est déjà 0 grâce au memset (champ réservé)
    packet[0] = 0x00;

    // ✅ CORRECTION 2 : emitter_id sur 1 octet (était memset incorrect)
    packet[1] = sender_id;

    // ✅ CORRECTION 3 : payload_length sur 2 octets en big-endian
    // (était memset(packet + 2, payload_len, 2) : incorrect, memset remplit
    // N octets avec la MÊME valeur, sans gestion big-endian)
    packet[2] = (payload_len >> 8) & 0xFF;  // octet de poids fort
    packet[3] = (payload_len)      & 0xFF;  // octet de poids faible

    // ✅ CORRECTION 4 : packet_serial sur 4 octets en big-endian
    // (était memset(packet + 4, serial, 4) : même problème qu'au-dessus)
    packet[4] = (serial >> 24) & 0xFF;  // octet de poids fort
    packet[5] = (serial >> 16) & 0xFF;
    packet[6] = (serial >> 8)  & 0xFF;
    packet[7] = (serial)       & 0xFF;  // octet de poids faible

    // Le tag est calculé sur l'en-tête + le payload
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH,
                packet,
                payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}


	// TO DO :  replace the two previous command by properly
	//			setting the packet header with the following structure :
	/***************************************************************************
	 *    Field       	Length (bytes)      Encoding        Description
	 ***************************************************************************
	 *  r 					1 								Reserved, set to 0.
	 * 	emitter_id 			1 					BE 			Unique id of the sensor node.
	 *	payload_length 		2 					BE 			Length of app_data (in bytes).
	 *	packet_serial 		4 					BE 			Unique and incrementing id of the packet.
	 *	app_data 			any 							The feature vectors.
	 *	tag 				16 								Message authentication code (MAC).
	 *
	 *	Note : BE refers to Big endian
	 *		 	Use the structure 	packet[x] = y; 	to set a byte of the packet buffer
	 *		 	To perform bit masking of the specific bytes you want to set, you can use
	 *		 		- bitshift operator (>>),
	 *		 		- and operator (&) with hex value, e.g.to perform 0xFF
	 *		 	This will be helpful when setting fields that are on multiple bytes.
	*/


	// For the tag field, you have to calculate the tag. The function call below is correct but
	// tag_cbc_mac function, calculating the tag, is not implemented.

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


void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {

    // Buffer 16 bytes (4 * 32 bits)
    uint32_t statew[4] = {0};
    // state is a pointer to the start of the buffer
    uint8_t *state = (uint8_t*) statew;
    size_t i;

    // Nombre de blocs (arrondi supérieur)
    size_t n = (msg_len + 15) / 16;

    // Cas message vide
    if (msg_len == 0) {
        for (int j = 0; j < 16; j++) {
            tag[j] = 0;
        }
        return;
    }

    for (i = 0; i < n; i++) {

        uint8_t block[16] = {0};

        // Taille réelle du bloc courant
        size_t remaining = msg_len - i * 16;
        size_t block_len = (remaining >= 16) ? 16 : remaining;

        // Copie + zero padding implicite
        for (int j = 0; j < block_len; j++) {
            block[j] = msg[i * 16 + j];
        }

        // XOR avec l'état précédent
        for (int j = 0; j < 16; j++) {
            state[j] ^= block[j];
        }

        // Chiffrement AES
        AES128_encrypt(state, AES_Key);
    }

    // Copie du tag
    for (int j = 0; j < 16; j++) {
        tag[j] = state[j];
    }
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

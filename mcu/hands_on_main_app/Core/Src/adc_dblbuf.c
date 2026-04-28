#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"




static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static volatile uint8_t ADCDataRdy[2] = {0, 0};

static volatile uint8_t cur_melvec = 0;
static q15_t mel_vectors[N_MELVECS][MELVEC_LENGTH];

static uint32_t packet_cnt = 0;

static volatile int32_t rem_n_bufs = 0;



int StartADCAcq(int32_t n_bufs) {
	rem_n_bufs = n_bufs;
	cur_melvec = 0;
	if (rem_n_bufs != 0) {
		return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
	} else {
		return HAL_OK;
	}
}

int IsADCFinished(void) {
	return (rem_n_bufs == 0);
}

static void StopADCAcq() {
	HAL_ADC_Stop_DMA(&hadc1);
}

static void print_spectrogram(void) {
#if (DEBUGP == 1)
	start_cycle_count();
	DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
	for(unsigned int j=0; j < N_MELVECS; j++) {
		DEBUG_PRINT("FV #%u:\t", j+1);
		for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
			DEBUG_PRINT("%.2f, ", q15_to_float(mel_vectors[j][i]));
		}
		DEBUG_PRINT("\r\n");
	}
	stop_cycle_count("Print FV");
#endif
}

static void print_encoded_packet(uint8_t *packet) {
#if (DEBUGP == 1)
	char hex_encoded_packet[2*PACKET_LENGTH+1];
	hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
	DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
#endif
}


/**
 * Convolutional encoder R=1/2, bit à bit
 * gn=[1,1], gd=[1,1,1]
 * OUTPUT est 2x plus long que INPUT
 *
 * @param input_bytes   payload original,       taille n_bytes
 * @param output_bytes  payload encodé,          taille 2*n_bytes
 * @param n_bytes       taille du payload original
 */
// void conv_encode_bitwise(const uint8_t *input_bytes, uint8_t *output_bytes, size_t n_bytes) {
//     uint8_t s1 = 0, s2 = 0;
//     size_t out_byte_idx = 0;
//     int out_bit_idx = 7;  // MSB first

//     memset(output_bytes, 0, 2 * n_bytes);

//     for (size_t byte_i = 0; byte_i < n_bytes; byte_i++) {
//         for (int bit_i = 7; bit_i >= 0; bit_i--) {
//             uint8_t u = (input_bytes[byte_i] >> bit_i) & 1;

//             // Feedback: gd=[1,1,1]
//             uint8_t v = u ^ s1 ^ s2;

//             // Sortie systématique: u lui-même
//             uint8_t sys = u;
//             // Sortie non-systématique: gn=[1,1]
//             uint8_t cod = v ^ s1;

//             // Écriture bit systématique
//             output_bytes[out_byte_idx] |= (sys << out_bit_idx);
//             out_bit_idx--;
//             if (out_bit_idx < 0) { out_bit_idx = 7; out_byte_idx++; }

//             // Écriture bit codé
//             output_bytes[out_byte_idx] |= (cod << out_bit_idx);
//             out_bit_idx--;
//             if (out_bit_idx < 0) { out_bit_idx = 7; out_byte_idx++; }

//             // Shift registre
//             s2 = s1;
//             s1 = v;
//         }
//     }
// }


// static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {
//     uint8_t *payload = packet + PACKET_HEADER_LENGTH;

//     // 1. Écriture des mel vectors
//     for (size_t i = 0; i < N_MELVECS; i++) {
//         for (size_t j = 0; j < MELVEC_LENGTH; j++) {
//             payload[(i*MELVEC_LENGTH+j)*2]   = mel_vectors[i][j] >> 8;
//             payload[(i*MELVEC_LENGTH+j)*2+1] = mel_vectors[i][j] & 0xFF;
//         }
//     }

//     // 2. Encodage convolutif → payload doublé
//     uint8_t encoded[2 * PAYLOAD_LENGTH];
//     conv_encode_bitwise(payload, encoded, PAYLOAD_LENGTH);
//     memcpy(payload, encoded, 2 * PAYLOAD_LENGTH);

//     // 3. Header + tag avec la nouvelle taille
//     make_packet(packet, 2 * PAYLOAD_LENGTH, 0, *packet_cnt);
//     *packet_cnt += 1;
// }

static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {
	// BE encoding of each mel coef
	
	for (size_t i=0; i<N_MELVECS; i++) {
		for (size_t j=0; j<MELVEC_LENGTH; j++) {
			(packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2]   = mel_vectors[i][j] >> 8;
			(packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2+1] = mel_vectors[i][j] & 0xFF;
		}
	}
	// Write header and tag into the packet.
	make_packet(packet, PAYLOAD_LENGTH, 0, *packet_cnt);
	*packet_cnt += 1;
	if (*packet_cnt == 0) {
		// Should not happen as packet_cnt is 32-bit and we send at most 1 packet per second.
		DEBUG_PRINT("Packet counter overflow.\r\n");
		Error_Handler();
	}
}

static void send_spectrogram() {
	uint8_t packet[PACKET_LENGTH];

	// start_cycle_count();
	encode_packet(packet, &packet_cnt);
	// stop_cycle_count("Encode packet");

	// start_cycle_count();
	S2LP_Send(packet, PACKET_LENGTH);
	// stop_cycle_count("Send packet");

	print_encoded_packet(packet);
}
int go = 0;
static void ADC_Callback(int buf_cplt) {
	if (rem_n_bufs != -1) {
		rem_n_bufs--;
	}
	if (rem_n_bufs == 0) {
		StopADCAcq();
	} else if (ADCDataRdy[1-buf_cplt]) {
		DEBUG_PRINT("Error: ADC Data buffer full\r\n");
		Error_Handler();
	}
	ADCDataRdy[buf_cplt] = 1;
	//start_cycle_count();
	Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
	static arm_rfft_instance_q15 rfft_inst;
	arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
	Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors[cur_melvec], rfft_inst);
	for(int i=0; i < N_SPECS*N_MELVECS-1; i++){
		long_sum_energy[i] = long_sum_energy[i+1];
	}
	Threshold = 0.0;
	long_sum_energy[N_SPECS*N_MELVECS-1] = energy_sum;
	for(int i=0; i < N_SPECS*N_MELVECS; i++){
		Threshold += long_sum_energy[i];
	}
	Threshold /= (N_SPECS*N_MELVECS);
	Threshold *= 2; // Adjust this multiplier based on your requirements
	if(sending){
		go = 1;
	}
	go = 1;
	cur_melvec++;
	//stop_cycle_count("spectrogram");
	ADCDataRdy[buf_cplt] = 0;
	if (rem_n_bufs == 0) {
		if(go){
			// print_spectrogram();
			send_spectrogram();
			go = 0;
		}else{			DEBUG_PRINT("Not sending packet\r\n");
		}
		// print_spectrogram();
		// send_spectrogram();
	}
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(1);
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(0);
}

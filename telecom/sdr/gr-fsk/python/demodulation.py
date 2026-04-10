#!/usr/bin/env python
#
# Copyright 2021 UCLouvain.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from distutils.version import LooseVersion

import numpy as np
from gnuradio import gr




def demodulate(y, B, R, Fdev):
    """
    Démodulateur non-cohérent vectorisé et ultra-rapide (FSK).
    """
    nb_syms = len(y) // R
    y = y[:nb_syms * R].reshape(nb_syms, R)  # Vue sans recopie

    # Références de phase
    ph = 2 * np.pi * Fdev * np.arange(R) / (R * B)
    s_0 = np.exp(-1j * ph)
    s_1 = np.exp( 1j * ph)

    # Corrélation vectorisée sur tout le signal
    r0 = np.abs(y @ s_0)  # Produit matriciel = somme sur R pour chaque symbole
    r1 = np.abs(y @ s_1)

    # Décision binaire vectorisée
    bits_hat = (r1 < r0).astype(np.uint8)

    return bits_hat


# def number2binary(x0, length):
#         binary_array = np.zeros((length,))
   
#         x = x0
#         i = 0
   
#         while x > 1 and i < length:
#             binary_array[i] = x % 2
#             x = int(x / 2)
#             i = i + 1
   
#         if x > 0 and i < length:
#             binary_array[i] = 1
   
#         return binary_array[::-1]
   
# def binary2number(x):
#     out = 0
#     for i in x:
#         out = 2*out + i
#     return out


# def viterbi_decoder(R1,R0,symb_R1,symb_R0,len_b,x_tilde):
#         def dist(a,b):
#             return np.abs(a-b)**2
   
#         N_b = int(len(x_tilde)/len_b)
   
#         x_tilde_b = np.reshape(x_tilde,(N_b,len_b))
#         u_hat_b = np.zeros(x_tilde_b.shape,dtype=np.int32)
   
#         nb_states = len(R1)

#         for i in range(N_b):          
#             x_tilde_i  = x_tilde_b[i,:]
#             u_hat_i = u_hat_b[i,:]
       
#             bits = np.zeros((nb_states,len_b))
#             weights = np.inf*np.ones((nb_states,))
#             weights[0] = 0
       
#             new_states = np.zeros((2,nb_states))
#             new_weights = np.zeros((2,nb_states))
#             new_bits = np.zeros((2,nb_states,len_b))  
       
#             for j in range(len_b):
#                 for k in range(nb_states):
#                     new_states[1,k] = R1[k]
#                     new_states[0,k] = R0[k]
#                     new_weights[1,k] = weights[k] + dist(x_tilde_i[j],symb_R1[k])
#                     new_weights[0,k] = weights[k] + dist(x_tilde_i[j],symb_R0[k])      
#                     new_bits[1,k,:] = bits[k,:]
#                     new_bits[0,k,:] = bits[k,:]
#                     new_bits[1,k,j] = 1
               
#                 for k in range(nb_states):
#                     idx_0_filled = False
#                     for l in range(nb_states):
#                         if new_states[0,l] == k:
#                             if idx_0_filled:
#                                 idx_10 = 0
#                                 idx_11 = l
#                             else:
#                                 idx_00 = 0
#                                 idx_01 = l
#                                 idx_0_filled = True
                           
#                         if new_states[1,l] == k:
#                             if idx_0_filled:
#                                 idx_10 = 1
#                                 idx_11 = l
#                             else:
#                                 idx_00 = 1
#                                 idx_01 = l
#                                 idx_0_filled = True
               
#                     if new_weights[idx_00,idx_01] <= new_weights[idx_10,idx_11]:
#                         weights[k] = new_weights[idx_00,idx_01]
#                         bits[k,:] = new_bits[idx_00,idx_01,:]
#                     else:
#                         weights[k] = new_weights[idx_10,idx_11]
#                         bits[k,:] = new_bits[idx_10,idx_11,:]

#             final_weight = np.inf
#             for k in range(nb_states):
#                 if weights[k] < final_weight:
#                     final_weight = weights[k]
#                     u_hat_i[:] = bits[k,:]
   
#         u_hat = np.reshape(u_hat_b,(u_hat_b.size,))
#         return u_hat

# def poly2trellis(gn,gd):
#         M = max(len(gn),len(gd)) - 1
#         nb_states = 2**M
   
#         alpha = np.zeros((M+1,))
#         beta = np.zeros((M+1,))
   
#         alpha[:len(gn)] = gn
#         beta[:len(gd)] = gd

#         R1 = np.zeros((nb_states,),dtype=np.int32)
#         R0 = np.zeros((nb_states,),dtype=np.int32)
   
#         out_R1 = np.zeros((nb_states,2),dtype=np.int32)
#         out_R0 = np.zeros((nb_states,2),dtype=np.int32)
   
#         out_R1[:,0] = 1
   
#         for i in range(nb_states):
#             states = np.zeros((M+1,))
#             states[:M] = number2binary(i,M)[::-1]
       
#             y_1 = (alpha[0] + states[0]) % 2
#             y_0 = states[0]
       
#             new_states_1 = (alpha[1:] + beta[1:]*y_1 + states[1:]) % 2
#             new_states_0 = (beta[1:]*y_0 + states[1:]) % 2
       
#             R1[i] = binary2number(new_states_1[::-1])
#             R0[i] = binary2number(new_states_0[::-1])
       
#             out_R1[i,1] = int(y_1)
#             out_R0[i,1] = int(y_0)
   
#         return R1,R0,out_R1,out_R0


# def demodulate(y, B, R, Fdev):
#         # """Non-coherent demodulator."""
   
#         # nb_syms = len(y) // R  # Number of CPFSK symbols in y
#         fd = Fdev  # Frequency deviation, Delta_f

#         """
#         Démodulateur non-cohérent vectorisé et ultra-rapide (FSK).
#         """
#         nb_syms = len(y) // R
#         y = y[:nb_syms * R].reshape(nb_syms, R)  # Vue sans recopie

#         # Références de phase
#         ph = 2 * np.pi * fd * np.arange(R) / (R * B)
#         s_0 = np.exp(-1j * ph)
#         s_1 = np.exp( 1j * ph)

#         # Corrélation vectorisée sur tout le signal
#         r0 = np.abs(y @ s_0)  # Produit matriciel = somme sur R pour chaque symbole
#         r1 = np.abs(y @ s_1)

#         # Décision binaire vectorisée
#         bits_hat = (r1 < r0).astype(np.uint8)

#         R1,R0,out_R1,out_R0 = poly2trellis(np.array([1, 1]), np.array([1, 1, 1]))
#         nb_states = len(R1)
#         symb_R1 = np.ones(nb_states, dtype=np.float64)   # symbole associé à bit=1
#         symb_R0 = np.zeros(nb_states, dtype=np.float64)  # symbole associé à bit=0
#         decoded_bits = viterbi_decoder(R1, R0, symb_R1, symb_R0, 1, bits_hat)
#         return decoded_bits






class demodulation(gr.basic_block):
    """
    docstring for block demodulation
    """

    def __init__(self, drate, fdev, fsamp, payload_len, crc_len):
        self.drate = drate
        self.fdev = fdev
        self.fsamp = fsamp
        self.frame_len = payload_len + crc_len
        self.osr = int(fsamp / drate)

        gr.basic_block.__init__(
            self, name="Demodulation", in_sig=[np.complex64], out_sig=[np.uint8]
        )

        self.gr_version = gr.version()

    def forecast(self, noutput_items, ninputs):
        """
        Forecast is only called from a general block
        this is the default implementation
        """
        ninput_items_required = [0] * ninputs
        for i in range(ninputs):
            ninput_items_required[i] = noutput_items * self.osr * 8

        return ninput_items_required

    def symbols_to_bytes(self, symbols):
        """
        Converts symbols (bits here) to bytes
        """
        if len(symbols) == 0:
            return []

        n_bytes = int(len(symbols) / 8)
        bitlists = np.array_split(symbols, n_bytes)
        out = np.zeros(n_bytes).astype(np.uint8)

        for i, l in enumerate(bitlists):
            for bit in l:
                out[i] = (out[i] << 1) | bit

        return out

    def general_work(self, input_items, output_items):
        n_syms = len(output_items[0]) * 8
        buf_len = n_syms * self.osr

        y = input_items[0][:buf_len]
        self.consume_each(buf_len)

        s = demodulate(y, self.drate, self.osr, self.fdev)
        b = self.symbols_to_bytes(s)
        output_items[0][: len(b)] = b

        return len(b)

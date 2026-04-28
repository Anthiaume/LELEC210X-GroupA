"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
import soundfile as sf
from serial.tools import list_ports
import pickle
import sys
import ctypes
import time
sys.path.append('classification/src/classification/utils/')
import audio_student

PRINT_PREFIX = "SND:HEX:"
#FREQ_SAMPLING = 10200
FREQ_SAMPLING=11000
VAL_MAX_ADC = 4096
VDD = 3.3
N_MELVECS=20
NUMBER="Number:"
FORMAT= "After Format"

def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
        return bytes.fromhex(line[len(SECOND_PREFIX):])
    elif line.startswith(NUMBER):
        print(line[len(NUMBER):])
    elif line.startswith(FORMAT):
       # print(line[len(FORMAT):])
        #print("type",type(line[len(FORMAT)+1]))
        #print(line[len(FORMAT):])
        return bytes.fromhex(line[len(FORMAT):])
    else:    
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1042).decode("ascii")
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder("<")
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array

def writer(port=None):

    ser = serial.Serial("COM6",115200)

    
    chunk = 512

   # audio= sf.read("mcu/hands_on_audio_acquisition/audio_files/chainsaw_07.wav")
    #audio = audio_student.AudioUtil.resample(audio)[0]
    #audio = audio.mean(axis=1) if audio.ndim>1 else audio
    #audio = audio / np.max(np.abs(audio))
    #audio = np.clip(audio, -1.0, 1.0)

    #audio = ((audio + 1.0) * 2047.5).astype(np.int16)
    audio = np.arange(0,N_MELVECS*chunk,1, dtype=np.int16)
    audio = (audio % 4096).astype(np.int16)
    """
    audio = audio.mean(axis=1) if audio.ndim>1 else audio
    audio = audio/np.max(np.abs(audio))
    audio = np.clip(audio, -1.0, 1.0)
    audio = ((audio + 1.0) * 2047.5).astype(np.uint16)
    """
    #audio = np.arange(N_MELVECS*chunk, dtype=np.uint16)
    print("audio :",audio)
    for i in range(0, N_MELVECS*chunk, chunk):
        #ser.write(audio[i:i+chunk].tobytes())
        ser.write(audio[i:i+chunk])
        time.sleep(0.02)
    
   
    time.sleep(1)
    print("Fin de la transmission")
    
    
    while True:
        line = ""
        print("avant while")
        while not line.endswith("\n"):
            
            line += ser.read_until(b"\n", size=20495).decode("ascii")
        print("après while")    
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            dt = np.dtype(np.int16)
            dt = dt.newbyteorder("<")
            buffer_array = np.frombuffer(buffer, dtype=dt)
           # print(buffer_array)
            yield buffer_array


def generate_audio(buf, file_name):
    buf = np.asarray(buf, dtype=np.float64)
    buf = buf - np.mean(buf)
    buf /= max(abs(buf))
    sf.write("audio_files/" + file_name + ".ogg", buf, FREQ_SAMPLING)
    print("je suis ici")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        plt.figure(figsize=(10, 5))
        input_stream = writer(port=args.port)
        msg_counter = 0
        print("ici")
        for msg in input_stream:
           # sf.write("audio_files/" + f"acq-{msg_counter}" + ".ogg", msg, FREQ_SAMPLING)
            print(f"Acquisition #{msg_counter}")
            print(msg)
            audio = np.arange(0,N_MELVECS*512,1, dtype=np.int16)
            vec = (audio % 4096).astype(np.int16)
            
            #y=audio_student.AudioUtil.specgram(sf.read("mcu/hands_on_audio_acquisition/audio_files/chainsaw_07.wav"))
            y = audio_student.AudioUtil.specgram(vec)
            
           # y=audio_student.AudioUtil.specgram(audio)
            #print(y[0][0],msg[0])
            #print("at index 5: ",y[10],msg[10])
            for i in range(len(y)):
                
                print("diff :",y[i]-msg[i])
            
            pickle.dump(msg, open(f"mcu/hands_on_audio_acquisition/audio_files/newtest_{msg_counter}.pkl", "wb"))


            """
            buffer_size = len(msg)
            times = np.linspace(0, buffer_size - 1, buffer_size) * 1 / FREQ_SAMPLING
            voltage_V = msg * VDD / VAL_MAX_ADC*1E3

            plt.plot(times, voltage_V)
            plt.title(f"Acquisition #{msg_counter}")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.ylim([0, 3.3])
            plt.draw()
            plt.pause(0.001)
            plt.cla()

            generate_audio(msg, f"acq-{msg_counter}")
            """
            msg_counter += 1

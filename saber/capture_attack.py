#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import sys
import time
import chipwhisperer as cw
from chipwhisperer.common.traces import Trace
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import fnmatch
from pathlib import Path
from logging import StreamHandler

sys.path.append("..")
from ECC_CCT_tool.ECC_CCT_tool import ECC_CCT_TOOL

###########################################

SETUP = "Masked Shuffled Saber"
REPROGRAM = True
START_CT = 0
CODE_DISTANCE = 6
GEN_KEYS = True
SEND_KEYS = True

########

SAVED_SETUPS = {
    "Masked Shuffled Saber": {
        "plot": False,
        "plot_type": "averaged",
        "hex_path": "./hexes/shuffled_masked_saber.hex",
        "short_delay": 0.1,
        "long_delay": 0.20,
        "repeat_traces": 3,
        "num_of_samples": 96000,
        "samples_per_cycle": "clkgen_x1",
        "decimate_samples": 1,
        "presamples": 200,
        "output_bytes": 288,
        },
}

###########################################

PLOT = SAVED_SETUPS[SETUP]["plot"]
PLOT_TYPE = SAVED_SETUPS[SETUP]["plot_type"]
HEX_PATH = SAVED_SETUPS[SETUP]["hex_path"]

SHORT_DELAY = SAVED_SETUPS[SETUP]["short_delay"]
LONG_DELAY = SAVED_SETUPS[SETUP]["long_delay"]

REPEAT_TRACES = SAVED_SETUPS[SETUP]["repeat_traces"]
NUM_OF_SAMPLES = SAVED_SETUPS[SETUP]["num_of_samples"]
SAMPLES_PER_CYCLE = SAVED_SETUPS[SETUP]["samples_per_cycle"]
DECIMATE_SAMPLES = SAVED_SETUPS[SETUP]["decimate_samples"]
PRESAMPLES = SAVED_SETUPS[SETUP]["presamples"]

PROJECT_NAME = "{}_x3_d{}_p{}_s{}".format(SETUP, DECIMATE_SAMPLES, PRESAMPLES, NUM_OF_SAMPLES)

OUTPUT_BYTES = SAVED_SETUPS[SETUP]["output_bytes"]

scope = cw.scope(type=cw.scopes.OpenADC)

target = cw.target(scope, cw.targets.SimpleSerial)

STM32_HSE = 24000000*3

scope.default_setup()
scope.clock.clkgen_freq = STM32_HSE
scope.clock.freq_ctr_src = "clkgen"
scope.clock.adc_src = "clkgen_x1"
scope.adc.samples = NUM_OF_SAMPLES
scope.adc.decimate = DECIMATE_SAMPLES
scope.adc.offset = 0
scope.adc.presamples = PRESAMPLES

time.sleep(0.25)

print(scope.clock.adc_freq, flush=True)
print(scope.clock.adc_rate, flush=True)
print(scope.clock.freq_ctr_src, flush=True)
print(scope.clock.clkgen_src, flush=True)
print(scope.clock.clkgen_mul, flush=True)
print(scope.clock.clkgen_div, flush=True)

print("Timeout is          :",scope.adc.timeout, flush=True)
print("Output Clock is     :",scope.clock.freq_ctr/1000000,"MHz", flush=True)
print("ADC Clock is        :",scope.clock.adc_freq/1000000,"MHz", flush=True)
print("Sampling Freq is    :",scope.clock.adc_freq/STM32_HSE, flush=True)
print("ADC_PLL locked      :",scope.clock.adc_locked, flush=True)
print("ADC Capture Samples :",scope.adc.samples, flush=True)
print("ADC Decimate        :",scope.adc.decimate, flush=True)
print("Trigger Pin         :",scope.trigger.triggers, flush=True)
print("Trigger States      :",scope.io.tio_states, flush=True)


if REPROGRAM:
    cw.program_target(scope, cw.programmers.STM32FProgrammer, HEX_PATH)

# ./capture_attack.py <set>
if len(sys.argv) == 2:
    TracePath = f"traces/attack/set_{sys.argv[1]}/"
else:
    TracePath = "traces/attack/"

keypath = TracePath+"GNDTruth/"

ECC_tool = ECC_CCT_TOOL("saber", CODE_DISTANCE)

target.output_len = OUTPUT_BYTES

msg = bytearray([1]*1)

GOT_WARNING = False

class Catch_handler(StreamHandler):
    def __init__(self):
        StreamHandler.__init__(self)

    def emit(self, record):
        global GOT_WARNING
        msg = self.format(record)
        print("Caught warning/error", flush=True)
        GOT_WARNING = True

def genKeyPair():
    print("Generate key-pair", flush=True)

    #generate a random keypair
    target.simpleserial_write('g', msg)


    # Ask stm32 to send Secret Key
    time.sleep(0.1)
    target.simpleserial_write('s', msg)

    sk = target.simpleserial_read('s', 64, timeout=1250,ack=False)
    time.sleep(0.01)
    for x in range(0,35):
        sk.extend(target.simpleserial_read('s', 64, end='\n', timeout=1250, ack=False))
        time.sleep(0.01)

    sk_file = open(keypath+"SecKey.bin", "wb")
    sk_file.write(sk)
    sk_file.close()

    # Ask stm32 to send Public Key

    target.simpleserial_write('t', msg)

    time.sleep(0.01)
    pk = target.simpleserial_read('t', 64, timeout=1250,ack=False)

    for x in range(0,14):
        pk.extend(target.simpleserial_read('t', 64, timeout=1250,ack=False))
        time.sleep(0.01)
    pk.extend(target.simpleserial_read('t', 32, timeout=1250,ack=False))

    pk_file = open(keypath+"PubKey.bin", "wb")
    pk_file.write(pk)
    pk_file.close()

    len(pk)

def sendkeypair(uploadkeypath):
    print("Send key-pair", flush=True)

    msg = bytearray([1]*1)

    #send key
    target.simpleserial_write('r', msg)
    time.sleep(0.1)
    # ---- read key if needed -----
    skfile = bytearray()
    with open(uploadkeypath+"/SecKey.bin", "rb") as f:
        byte = f.read(1)
        while byte != b"":
            # Do stuff with byte.
            skfile.extend(byte)
            byte = f.read(1)
    len(skfile)
    # -----------------------------

    for x in range(0, 2304, 64):
        arrbuffer = bytearray()
        arrbuffer.extend(skfile[x:(x+64)])
        target.simpleserial_write('k', arrbuffer)
        time.sleep(0.1) # Computer begins next transmission too fast for stm32

    # Transmit Public Key from keyfile to STM32

    pkfile = bytearray()
    with open(uploadkeypath+"/PubKey.bin", "rb") as f:
        byte = f.read(1)
        while byte != b"":
            # Do stuff with byte.
            pkfile.extend(byte)
            byte = f.read(1)
    len(pkfile)

    for x in range(0, 992, 32):
        arrbuffer = bytearray()
        arrbuffer.extend(pkfile[x:(x+32)])
        target.simpleserial_write('m', arrbuffer)
        time.sleep(0.1) # Computer begins next transmission too fast for stm32

    # Ask stm32 to send Secret Key
    time.sleep(0.1)
    target.simpleserial_write('s', msg)

    sk = target.simpleserial_read('s', 64, timeout=1250,ack=False)
    time.sleep(0.01)
    for x in range(0,35):
        sk.extend(target.simpleserial_read('s', 64, end='\n', timeout=1250, ack=False))
        time.sleep(0.01)

    print("Secret key successfully uploaded:", skfile==sk, flush=True)
    if not skfile==sk:
        sys.exit()

    # Ask stm32 to send Public Key

    time.sleep(0.1)
    target.simpleserial_write('t', msg)

    pk = target.simpleserial_read('t', 64, timeout=1250,ack=False)
    time.sleep(0.01)
    for x in range(0,14):
        pk.extend(target.simpleserial_read('t', 64, timeout=1250,ack=False))
    time.sleep(0.01)
    pk.extend(target.simpleserial_read('t', 32, timeout=1250,ack=False))

    print("Public key successfully uploaded:", pkfile==pk, flush=True)
    if not pkfile==pk:
        sys.exit()

    time.sleep(0.1)

def capture_trace_kalle(scope, target, plaintext, key=None, ack=True):
    import signal, logging

    # useful to delay keyboard interrupt here,
    # since could interrupt a USB operation
    # and kill CW until unplugged+replugged
    class DelayedKeyboardInterrupt:
        def __enter__(self):
            self.signal_received = False
            self.old_handler = signal.signal(signal.SIGINT, self.handler)

        def handler(self, sig, frame):
            self.signal_received = (sig, frame)
            logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

        def __exit__(self, type, value, traceback):
            signal.signal(signal.SIGINT, self.old_handler)
            if self.signal_received:
                self.old_handler(*self.signal_received)
    with DelayedKeyboardInterrupt():
        if key:
            target.set_key(key, ack=ack)

        scope.arm()

        if plaintext:
            target.simpleserial_write('p', plaintext)

        ret = scope.capture()

        i = 0
        while not target.is_done():
            i += 1
            time.sleep(0.05)
            if i > 100:
                warnings.warn("Target did not finish operation")
                return None

        if ret:
            warnings.warn("Timeout happened during capture")
            return None

        response = target.simpleserial_read('r', target.output_len, ack=ack,timeout=2500)
        wave = scope.get_last_trace()

    if len(wave) >= 1:
        return Trace(wave, plaintext, response, key)
    else:
        return None

def getTraces():
    global GOT_WARNING

    file_number = -1
    for file_name in os.listdir(TracePath):
        if fnmatch.fnmatch(file_name, "shuffle_traces_part[0-3]_[0-9]*.npy"):
            n = int(file_name[21:-4])
            if n > file_number: file_number = n
    file_number += 1
    print("file_number =", file_number, flush=True)

    catch_warnings = Catch_handler()
    #catch_warnings.setLevel(logging.DEBUG)
    cw.target_logger.addHandler(catch_warnings)
            
    for part in range(0,3):
        #print("Part", part)

        for CT_num, CTset in enumerate(ECC_tool.ct_table):
            #print("CT", CT_num)

            if part*len(ECC_tool.ct_table)+CT_num < START_CT: continue

            shuffle_traces = np.zeros(shape=(128*REPEAT_TRACES, 41500))
            message_traces = np.zeros(shape=(128*REPEAT_TRACES, 55100))
            shuffle_labels = np.zeros(shape=(128*REPEAT_TRACES, 256))
            message_labels = np.zeros(shape=(128*REPEAT_TRACES, 32))

            with tqdm(total=128, desc=f"Capturing part:{part+1}/3 CT-set:{CT_num}") as pbar:

                for rot_num in range(1,256,2):

                    ct = ECC_tool.CCT(CTset, part, rot_num, [x for x in range(1,255)])
                
                    target.simpleserial_write('o', msg)
                    time.sleep(0.1)
                    for x in range(0, 1088, 64):
                        temp = ct[x:x+64]
                        target.simpleserial_write('j', temp)
                        time.sleep(SHORT_DELAY) # Computer begins next transmission too fast for stm32
                        
                    index_offset = int((rot_num-1)/2*REPEAT_TRACES)
                    for j in range(REPEAT_TRACES):
                        try:
                            while True:
                                trace = capture_trace_kalle(scope, target, msg, ack=False)
                                trig = scope.adc.trig_count
                                time.sleep(LONG_DELAY)
                                if trig >= 40850 or trig <= 40770: continue
                                if trace == None: continue
                                if trace.textout == None: continue
                                if GOT_WARNING:
                                    print("Resetting warning", flush=True)
                                    GOT_WARNING = False
                                    continue
                                shuffle_traces[index_offset+j] = trace.wave[:41500]
                                message_traces[index_offset+j] = trace.wave[trig-100:trig+55000]
                                shuffle_labels[index_offset+j] = trace.textout[:256]
                                message_labels[index_offset+j] = trace.textout[256:]
                                break
                        except Exception as e:
                            print("Exception occured for trace", j, flush=True)
                            raise(e)
                    pbar.update(1)
        
            #print("Saving traces")
            np.save(TracePath+f"shuffle_traces_part{part}_CT{CTset[0]}-{CTset[1]}_{file_number}", shuffle_traces)
            np.save(TracePath+f"message_traces_part{part}_CT{CTset[0]}-{CTset[1]}_{file_number}", message_traces)
            np.save(TracePath+f"shuffle_labels_part{part}_CT{CTset[0]}-{CTset[1]}_{file_number}", shuffle_labels.astype('int'))
            np.save(TracePath+f"message_labels_part{part}_CT{CTset[0]}-{CTset[1]}_{file_number}", message_labels.astype('int'))

    target.close()
        
if __name__ == "__main__":

    try:
        os.makedirs(keypath)    
        print("Directory " , keypath ,  " Created ", flush=True)
    except FileExistsError:
        print("Directory " , keypath ,  " already exists", flush=True)  

    if GEN_KEYS:
        genKeyPair()
    if SEND_KEYS:
        sendkeypair(keypath)
    genCipherText()
    getTraces()

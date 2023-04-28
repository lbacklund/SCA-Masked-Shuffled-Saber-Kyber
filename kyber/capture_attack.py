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

SETUP = "Masked Shuffled Kyber"
REPROGRAM = True
START_CT = 0
CODE_DISTANCE = 8
GEN_KEYS = True

########

SAVED_SETUPS = {
    "Masked Shuffled Kyber": {
        "hex_path": "./hexes/bit_shuffled_masked_kyber.hex",
        "short_delay": 0.09,
        "long_delay": 0.15,
        "repeat_traces": 30,
        "num_of_samples": 96000,
        "samples_per_cycle": "clkgen_x1",
        "decimate_samples": 1,
        "presamples": 200,
        "output_bytes": 288,
        },
}


###########################################

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
STM32_HSE = 24000000

scope.default_setup()
scope.clock.clkgen_freq = STM32_HSE
scope.clock.freq_ctr_src = "clkgen"
scope.clock.adc_src = "clkgen_x1"
scope.adc.samples = NUM_OF_SAMPLES
scope.adc.decimate = DECIMATE_SAMPLES
scope.adc.offset = 0
scope.adc.presamples = PRESAMPLES

time.sleep(0.25)

print(scope.clock.adc_freq)
print(scope.clock.adc_rate)
print(scope.clock.freq_ctr_src)
print(scope.clock.clkgen_src)
print(scope.clock.clkgen_mul)
print(scope.clock.clkgen_div)

print("Timeout is          :",scope.adc.timeout)
print("Output Clock is     :",scope.clock.freq_ctr/1000000,"MHz")
print("ADC Clock is        :",scope.clock.adc_freq/1000000,"MHz")
print("Sampling Freq is    :",scope.clock.adc_freq/STM32_HSE)
print("ADC_PLL locked      :",scope.clock.adc_locked)
print("ADC Capture Samples :",scope.adc.samples)
print("ADC Decimate        :",scope.adc.decimate)
print("Trigger Pin         :",scope.trigger.triggers)
print("Trigger States      :",scope.io.tio_states)


if REPROGRAM:
    cw.program_target(scope, cw.programmers.STM32FProgrammer, HEX_PATH)

# ./capture_attack.py <set>
if len(sys.argv) == 2:
    TracePath = f"traces/attack/set_{sys.argv[1]}/"
else:
    TracePath = "traces/attack/"

keypath = TracePath+"GNDTruth/"

ECC_tool = ECC_CCT_TOOL("kyber768", CODE_DISTANCE)

target.output_len = OUTPUT_BYTES

msg = bytearray([1]*1)


GOT_WARNING = False

class Catch_handler(StreamHandler):
    def __init__(self):
        StreamHandler.__init__(self)

    def emit(self, record):
        global GOT_WARNING
        msg = self.format(record)
        print("Caught warning/error")
        GOT_WARNING = True

def genKeyPair():
    print("Generate key-pair")

    #generate a random keypair
    target.simpleserial_write('g', msg)


    # Ask stm32 to send Secret Key
    time.sleep(0.1)
    target.simpleserial_write('s', msg)

    sk = target.simpleserial_read('s', 64, timeout=1250,ack=False)
    time.sleep(0.01)
    for x in range(0,11):
        sk.extend(target.simpleserial_read('s', 64, end='\n', timeout=1250, ack=False))
        time.sleep(0.01)

    coeffs = []
    for i in range(768):
        c = int(sk[i])
        c_bin = f"{c:08b}"
        if int(c_bin[0]):
            c = int(c_bin[1:], 2) - 128
        else:
            c = int(c_bin[1:], 2)
        coeffs.append(c)
    coeffs = np.array(coeffs)
    
    print(coeffs)

    np.save(keypath+"SecKey", coeffs)


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
    print("file_number =", file_number)

    catch_warnings = Catch_handler()
    #catch_warnings.setLevel(logging.DEBUG)
    cw.target_logger.addHandler(catch_warnings)
            
    for part in range(0,3):
        #print("Part", part)

        for CT_num, CTset in enumerate(ECC_tool.ct_table):
            #print("CT", CT_num)

            if part*len(ECC_tool.ct_table)+CT_num < START_CT: continue

            shuffle_traces = np.zeros(shape=(128*REPEAT_TRACES, 14000))
            message_traces = np.zeros(shape=(128*REPEAT_TRACES, 59000))
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
                        time.sleep(SHORT_DELAY)
                        
                    index_offset = int((rot_num-1)/2*REPEAT_TRACES)
                    for j in range(REPEAT_TRACES):
                        try:
                            while True:
                                trace = capture_trace_kalle(scope, target, msg, ack=False)
                                trig = scope.adc.trig_count
                                time.sleep(LONG_DELAY)
                                if trig >= 13615 or trig <= 13595: continue
                                if trace == None: continue
                                if trace.textout == None: continue
                                if GOT_WARNING:
                                    print("Resetting warning")
                                    GOT_WARNING = False
                                    continue
                                shuffle_traces[index_offset+j] = trace.wave[:14000]
                                message_traces[index_offset+j] = trace.wave[trig+7500:trig+66500]
                                shuffle_labels[index_offset+j] = trace.textout[:256]
                                message_labels[index_offset+j] = trace.textout[256:]
                                break
                        except Exception as e:
                            print("Exception occured for trace", j)
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
        print("Directory " , keypath ,  " Created ")
    except FileExistsError:
        print("Directory " , keypath ,  " already exists")  

    if GEN_KEYS:
        genKeyPair()
    getTraces()

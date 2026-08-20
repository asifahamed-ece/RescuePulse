# decode_raw_dump.py
import sys, numpy as np, soundfile as sf

hexstr = open(sys.argv[1]).read().strip()  # paste one hex line between BEGIN/END
raw = bytes.fromhex(hexstr)
pcm = np.frombuffer(raw, dtype='<u2').astype(np.int16)
sf.write("live_block.wav", pcm, 16000, subtype='PCM_16')
print(f"{len(pcm)} samples -> live_block.wav")
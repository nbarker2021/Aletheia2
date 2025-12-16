
import subprocess, pathlib, time, threading, os, sys
from scene8_adapter import FrameGenerator

STREAM_DIR = pathlib.Path('stream')
STREAM_DIR.mkdir(exist_ok=True)
# remove old segments
for f in STREAM_DIR.glob('*'):
    f.unlink()

ff_cmd = [
    'ffmpeg','-y',
    '-f','rawvideo','-pix_fmt','rgb24','-s','320x240','-r','30','-i','-',
    '-c:v','libx264','-preset','veryfast','-tune','zerolatency','-g','60',
    '-f','hls','-hls_time','2','-hls_list_size','5','-hls_flags','delete_segments',
    str(STREAM_DIR/'index.m3u8')
]
ff = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

gen = FrameGenerator(320,240)
try:
    for frame in gen:
        ff.stdin.write(frame.tobytes())
except KeyboardInterrupt:
    pass
finally:
    ff.stdin.close()
    ff.wait()

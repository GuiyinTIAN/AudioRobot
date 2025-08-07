# audio_recorder.py
import pyaudio
import wave
import threading

def record_with_pyaudio(filename, rate=16000, channels=1, chunk=1024):
    stop_event = threading.Event()
    frames = []

    def _record():
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=channels,
                         rate=rate,
                         input=True,
                         frames_per_buffer=chunk)
        print("Recording...")
        while not stop_event.is_set():
            data = stream.read(chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    print("Press Enter to start recording, then press Enter again to stop.")
    input()
    t = threading.Thread(target=_record)
    t.start()
    input()
    stop_event.set()
    t.join()

    pa = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Recording is saved:{filename}")
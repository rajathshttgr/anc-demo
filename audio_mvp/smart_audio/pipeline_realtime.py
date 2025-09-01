
import numpy as np

def run_realtime(sr=16000, block=1024, anc_type="nlms", filter_len=256, mu=0.3,
                 enhancement="wiener", ref_mic_index=None, pri_mic_index=None,
                 output_device_index=None):
    """Real-time demo using PyAudio: captures two mono inputs (reference + primary)
       and outputs processed audio with low latency.
       NOTE: You'll need an audio interface that supports 2 input channels or two mics; 
       or you can simulate reference from primary (less effective).
    """
    import pyaudio
    from .filters.anc import LMSANC, NLMSANC
    from .enh.classical import spectral_subtraction, wiener_filter

    if anc_type == "lms":
        anc = LMSANC(filter_len=filter_len, mu=mu*1e-3)
    else:
        anc = NLMSANC(filter_len=filter_len, mu=mu)

    pa = pyaudio.PyAudio()

    # Single-input fallback: duplicate primary as reference (not ideal)
    channels_in = 2 if ref_mic_index is not None else 1

    stream_in = pa.open(format=pyaudio.paFloat32, channels=channels_in, rate=sr,
                        input=True, frames_per_buffer=block,
                        input_device_index=pri_mic_index)

    stream_out = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr,
                         output=True, frames_per_buffer=block,
                         output_device_index=output_device_index)

    import collections
    overlap_buf = np.zeros(0, dtype=np.float32)

    try:
        while True:
            data = stream_in.read(block, exception_on_overflow=False)
            buf = np.frombuffer(data, dtype=np.float32)

            if channels_in == 2:
                buf = buf.reshape(-1, 2)
                primary = buf[:,0]
                reference = buf[:,1]
            else:
                primary = buf
                reference = primary  # fallback

            e, _ = anc.process(reference, primary)

            # Simple frame-wise enhancement
            if enhancement == "wiener":
                from .enh.classical import wiener_filter
                out = wiener_filter(e, sr)
            elif enhancement == "spectral_sub":
                from .enh.classical import spectral_subtraction
                out = spectral_subtraction(e, sr)
            else:
                out = e

            stream_out.write(out.astype(np.float32).tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        stream_in.stop_stream(); stream_in.close()
        stream_out.stop_stream(); stream_out.close()
        pa.terminate()

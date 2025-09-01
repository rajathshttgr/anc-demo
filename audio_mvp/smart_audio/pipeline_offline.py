
import numpy as np
from .utils.io_utils import load_audio, save_wav
from .filters.anc import LMSANC, NLMSANC
from .enh.classical import spectral_subtraction, wiener_filter

def align_ref_primary(primary, reference, delay_samples=0):
    """Optionally delay the reference to roughly align paths (tune empirically)."""
    if delay_samples > 0:
        reference = np.pad(reference, (delay_samples,0))[:len(primary)]
    reference = reference[:len(primary)]
    primary = primary[:len(reference)]
    return primary, reference

def offline_process(primary_path, reference_path=None, sr=16000, anc_type="nlms",
                    filter_len=256, mu=0.3, ref_delay=0, enhancement="wiener",
                    out_prefix="out"):
    """Offline pipeline: primary= speech+noise; reference= noise-correlated (optional).
       If no reference provided, ANC is skipped and only enhancement is applied.
    """
    y, fs = load_audio(primary_path, target_sr=sr)
    if reference_path:
        x, _ = load_audio(reference_path, target_sr=sr)
        y, x = align_ref_primary(y, x, delay_samples=ref_delay)
        if anc_type == "lms":
            anc = LMSANC(filter_len=filter_len, mu=mu*1e-3)
        else:
            anc = NLMSANC(filter_len=filter_len, mu=mu)
        e, _ = anc.process(x, y)
        anc_out = e
    else:
        anc_out = y

    if enhancement == "wiener":
        enh_out = wiener_filter(anc_out, sr)
    elif enhancement == "spectral_sub":
        enh_out = spectral_subtraction(anc_out, sr)
    else:
        enh_out = anc_out

    save_wav(f"{out_prefix}_anc.wav", anc_out, sr)
    save_wav(f"{out_prefix}_enh.wav", enh_out, sr)
    return y, anc_out, enh_out, sr

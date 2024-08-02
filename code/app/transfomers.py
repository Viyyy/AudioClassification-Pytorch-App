from torchaudio import transforms as T

def get_melspec_transformer(
    sample_rate, n_fft, n_mels, hop_length, win_length=None, **kwargs
):
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        # onesided=True,#  Argument 'onesided' has been deprecated and has no influence on the behavior of this module.
        n_mels=n_mels,
        mel_scale="htk",
        **kwargs
    )
from pathlib import Path
from yt_dlp import YoutubeDL
from pydub import AudioSegment


def convert_to_wav(file_path: str, output_dir: Path) -> str:
    input_file = Path(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{input_file.stem}.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    return str(wav_path)


def download_audio(url: str, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = Path(ydl.prepare_filename(info))

    wav_file = downloaded_file.with_suffix(".wav")
    audio = AudioSegment.from_file(wav_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_file, format="wav")
    return str(wav_file)


def chunk_audio(wav_path: str, output_dir: Path, chunk_min: int = 10) -> list[Path]:
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_min * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = chunks_dir / f"chunk_{i}.wav"
        audio[start:start + chunk_ms].export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_audio(source: str, work_dir: Path | None = None, chunk_min: int = 10) -> list[Path]:
    work_dir = work_dir or Path("downloads")
    work_dir.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        wav_path = download_audio(source, work_dir)
    else:
        wav_path = convert_to_wav(source, work_dir)
    return chunk_audio(wav_path, work_dir, chunk_min=chunk_min)

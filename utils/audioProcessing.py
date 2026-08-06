from pathlib import Path
from yt_dlp import YoutubeDL
from pydub import AudioSegment

def convert_to_wav(file_path: str) -> str:
    
    input_file = Path(file_path)
    output_dir = Path("downloads")
    output_dir.mkdir(exist_ok=True)
    wav_path = output_dir / f"{input_file.stem}.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")

    return str(wav_path)


def download_audio(url: str) -> str:
    
    output_dir = Path("downloads")
    output_dir.mkdir(exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }
                        ],
        "quiet": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = Path(ydl.prepare_filename(info))
    wav_file = downloaded_file.with_suffix(".wav")
    audio = AudioSegment.from_file(wav_file)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    wav_file = downloaded_file.with_suffix(".wav")
    audio.export(wav_file, format="wav")

    # downloaded_file.unlink()

    return str(wav_file)
    

def chunk_audio(wav_path:str,chunk_min:int = 10)->list:
    
    chunks_dir = Path("downloads/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    audio=AudioSegment.from_wav(wav_path)
    chunk_ms=chunk_min*60*1000
    chunks=[]
    for i,start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path=chunks_dir/f"chunk_{i}.wav"
        chunks.append(chunk_path)
        chunk.export(chunk_path, format="wav")
        
    return chunks


""" this is the main function"""

def process_audio(source: str,chunk_min: int = 10,) -> list[Path]:
    
    if source.startswith("http://") or source.startswith("https://"):
        wav_path=download_audio(source)
    else:
        wav_path=convert_to_wav(source)
        
    chunk_paths = chunk_audio(
        wav_path=wav_path,
        chunk_min=chunk_min
    )

    return chunk_paths
        
if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    wav_path = download_audio(url)
    print(f"\nSaved WAV file:\n{wav_path}")
    chunks=chunk_audio(wav_path=wav_path)
    print(chunks)
    
#url=https://www.youtube.com/watch?v=T-D1OfcDW1M
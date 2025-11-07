import argparse
import logging
import multiprocessing as mp
import os
import re
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from diarization import MSDDDiarizer
from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mp.set_start_method("spawn", force=True)

mtypes = {"cpu": "int8", "cuda": "float16"}


class MetricsCollector:
    """Collects and logs metrics for each processing step."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = []
        self.current_file_metrics = {}
        
    def start_file(self, audio_path: str):
        """Initialize metrics for a new file."""
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        self.current_file_metrics = {
            "file": str(audio_path),
            "file_size_mb": round(file_size, 2),
            "start_time": time.time(),
            "steps": {}
        }
        
    def record_step(self, step_name: str, duration: float):
        """Record timing for a processing step."""
        self.current_file_metrics["steps"][step_name] = round(duration, 2)
        
    def finish_file(self, success: bool, error: Optional[str] = None):
        """Finalize metrics for the current file."""
        total_time = time.time() - self.current_file_metrics["start_time"]
        self.current_file_metrics["total_time"] = round(total_time, 2)
        self.current_file_metrics["success"] = success
        if error:
            self.current_file_metrics["error"] = error
        self.metrics.append(self.current_file_metrics.copy())
        self._write_log(None)
        
    def _write_log(self, summary: Optional[Dict] = None):
        """Write metrics to log file."""
        log_data = {
            "run_timestamp": datetime.now().isoformat(),
            "files_processed": len(self.metrics),
            "metrics": self.metrics
        }
        if summary:
            log_data["summary"] = summary
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics:
            return {}
        
        successful = [m for m in self.metrics if m.get("success", False)]
        failed = [m for m in self.metrics if not m.get("success", False)]
        
        total_time = sum(m["total_time"] for m in self.metrics)
        avg_time = total_time / len(self.metrics) if self.metrics else 0
        
        step_times = {}
        for metric in successful:
            for step, duration in metric.get("steps", {}).items():
                if step not in step_times:
                    step_times[step] = []
                step_times[step].append(duration)
        
        step_avg = {step: round(sum(times) / len(times), 2) 
                   for step, times in step_times.items()}
        
        return {
            "total_files": len(self.metrics),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_seconds": round(total_time, 2),
            "average_time_per_file": round(avg_time, 2),
            "average_step_times": step_avg
        }


def diarize_parallel(audio: torch.Tensor, device, queue: mp.Queue):
    """Run diarization in a separate process."""
    model = MSDDDiarizer(device=device)
    result = model.diarize(audio)
    queue.put(result)


def process_single_file(
    audio_path: Path,
    output_dir: Optional[Path],
    metrics: MetricsCollector,
    whisper_model: faster_whisper.WhisperModel,
    whisper_pipeline: faster_whisper.BatchedInferencePipeline,
    alignment_model: torch.nn.Module,
    alignment_tokenizer,
    punct_model: Optional[PunctuationModel],
    device: str,
    language: Optional[str],
    suppress_tokens: List[int],
    batch_size: int,
    stemming: bool,
    model_name: str,
) -> bool:
    """
    Process a single audio file through the full pipeline.
    Returns True if successful, False otherwise.
    """
    try:
        metrics.start_file(str(audio_path))
        
        # Step 1: Source separation (if enabled)
        step_start = time.time()
        pid = os.getpid()
        temp_outputs_dir = f"temp_outputs_{pid}_{int(time.time())}"
        temp_path = os.path.join(os.getcwd(), temp_outputs_dir)
        os.makedirs(temp_path, exist_ok=True)
        
        if stemming:
            return_code = os.system(
                f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{temp_outputs_dir}" --device "{device}"'
            )
            
            if return_code != 0:
                logging.warning(
                    f"Source splitting failed for {audio_path}, using original audio file."
                )
                vocal_target = str(audio_path)
            else:
                vocal_target = os.path.join(
                    temp_outputs_dir,
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio_path))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = str(audio_path)
        
        metrics.record_step("source_separation", time.time() - step_start)
        
        # Step 2: Decode audio
        step_start = time.time()
        audio_waveform = faster_whisper.decode_audio(vocal_target)
        metrics.record_step("audio_decode", time.time() - step_start)
        
        # Step 3: Parallel transcription and diarization
        step_start = time.time()
        results_queue = mp.Queue()
        nemo_process = mp.Process(
            target=diarize_parallel,
            args=(
                torch.from_numpy(audio_waveform).unsqueeze(0),
                device,
                results_queue,
            ),
        )
        nemo_process.start()
        
        # Transcription (runs in parallel with diarization)
        transcribe_start = time.time()
        if batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=batch_size,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )
        transcription_time = time.time() - transcribe_start
        
        full_transcript = "".join(segment.text for segment in transcript_segments)
        
        # Wait for diarization to complete
        nemo_process.join()
        if results_queue.empty():
            raise RuntimeError("Diarization process did not return any results.")
        
        speaker_ts = results_queue.get_nowait()
        diarization_time = time.time() - step_start - transcription_time
        
        metrics.record_step("transcription", transcription_time)
        metrics.record_step("diarization", diarization_time)
        
        # Step 4: Forced alignment
        step_start = time.time()
        emissions, stride = generate_emissions(
            alignment_model,
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device),
            batch_size=batch_size,
        )
        
        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[info.language],
        )
        
        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )
        
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        metrics.record_step("forced_alignment", time.time() - step_start)
        
        # Step 5: Word-speaker mapping
        step_start = time.time()
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        metrics.record_step("word_speaker_mapping", time.time() - step_start)
        
        # Step 6: Punctuation restoration
        step_start = time.time()
        if info.language in punct_model_langs and punct_model:
            words_list = list(map(lambda x: x["word"], wsm))
            labled_words = punct_model.predict(words_list, chunk_size=230)
            
            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
            
            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
        else:
            if info.language not in punct_model_langs:
                logging.warning(
                    f"Punctuation restoration is not available for {info.language} language."
                )
        metrics.record_step("punctuation_restoration", time.time() - step_start)
        
        # Step 7: Realignment and sentence mapping
        step_start = time.time()
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
        metrics.record_step("realignment", time.time() - step_start)
        
        # Step 8: Write output files
        step_start = time.time()
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = audio_path.stem
            txt_path = output_dir / f"{base_name}.txt"
            srt_path = output_dir / f"{base_name}.srt"
        else:
            txt_path = audio_path.parent / f"{audio_path.stem}.txt"
            srt_path = audio_path.parent / f"{audio_path.stem}.srt"
        
        with open(txt_path, "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)
        
        with open(srt_path, "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)
        
        metrics.record_step("file_writing", time.time() - step_start)
        
        # Cleanup
        cleanup(temp_path)
        
        metrics.finish_file(success=True)
        return True
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing {audio_path}: {error_msg}", exc_info=True)
        metrics.finish_file(success=False, error=error_msg)
        # Cleanup on error
        if 'temp_path' in locals():
            try:
                cleanup(temp_path)
            except:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Bulk process audio files for transcription and diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output arguments
    parser.add_argument(
        "-d", "--directory",
        help="Directory containing audio files (default: current directory)",
        default=".",
        type=str,
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively in subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: same as input files)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log-file",
        help="Path to metrics log file (default: bulk_diarize_metrics.json)",
        type=str,
        default="bulk_diarize_metrics.json",
    )
    
    # Processing options
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation. "
        "This helps with long files that don't contain a lot of music.",
    )
    parser.add_argument(
        "--suppress-numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits. "
        "This helps the diarization accuracy but converts all digits into written text.",
    )
    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=4,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bulk_diarize.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize metrics collector
    metrics = MetricsCollector(args.log_file)
    
    # Find audio files
    audio_extensions = ['wav', 'mp3', 'm4a', 'opus', 'flac', 'ogg', 'aac', 'wma']
    search_path = Path(args.directory)
    audio_files = []
    
    if args.recursive:
        for ext in audio_extensions:
            audio_files.extend(search_path.rglob(f"*.{ext}"))
    else:
        for ext in audio_extensions:
            audio_files.extend(search_path.glob(f"*.{ext}"))
    
    if not audio_files:
        logging.warning(f"No audio files found in {args.directory}")
        return
    
    # Remove duplicates and sort
    audio_files = sorted(list(set(audio_files)))
    
    logging.info(f"Found {len(audio_files)} audio file(s) to process")
    logging.info(f"Using device: {args.device}")
    logging.info(f"Whisper model: {args.model_name}")
    
    # Process language argument
    language = process_language_arg(args.language, args.model_name)
    
    # Load models once (optimization for bulk processing)
    logging.info("Loading models...")
    model_load_start = time.time()
    
    # Load Whisper model
    whisper_model = faster_whisper.WhisperModel(
        args.model_name, device=args.device, compute_type=mtypes[args.device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if args.suppress_numerals
        else [-1]
    )
    
    # Load alignment model
    alignment_model, alignment_tokenizer = load_alignment_model(
        args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    
    # Load punctuation model (if needed, will check per file)
    punct_model = None
    try:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
    except Exception as e:
        logging.warning(f"Could not load punctuation model: {e}")
    
    model_load_time = time.time() - model_load_start
    logging.info(f"Models loaded in {model_load_time:.2f} seconds")
    
    # Process output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Process each file
    overall_start = time.time()
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        logging.info(f"\n[{i}/{len(audio_files)}] Processing: {audio_file}")
        
        success = process_single_file(
            audio_file,
            output_dir,
            metrics,
            whisper_model,
            whisper_pipeline,
            alignment_model,
            alignment_tokenizer,
            punct_model,
            args.device,
            language,
            suppress_tokens,
            args.batch_size,
            args.stemming,
            args.model_name,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Clear GPU cache between files
        if args.device == "cuda":
            torch.cuda.empty_cache()
    
    overall_time = time.time() - overall_start
    
    # Print summary
    summary = metrics.get_summary()
    summary["overall_time_seconds"] = round(overall_time, 2)
    summary["overall_time_minutes"] = round(overall_time / 60, 2)
    
    # Write final log with summary
    metrics._write_log(summary)
    
    logging.info("\n" + "="*60)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*60)
    logging.info(f"Total files processed: {summary['total_files']}")
    logging.info(f"Successful: {summary['successful']}")
    logging.info(f"Failed: {summary['failed']}")
    logging.info(f"Total processing time: {summary['overall_time_seconds']:.2f} seconds ({summary['overall_time_minutes']:.2f} minutes)")
    logging.info(f"Average time per file: {summary['average_time_per_file']:.2f} seconds")
    logging.info("\nAverage step times:")
    for step, avg_time in summary['average_step_times'].items():
        logging.info(f"  {step}: {avg_time:.2f} seconds")
    logging.info(f"\nDetailed metrics saved to: {args.log_file}")
    logging.info("="*60)
    
    # Cleanup models
    del whisper_model, whisper_pipeline, alignment_model
    if punct_model:
        del punct_model
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


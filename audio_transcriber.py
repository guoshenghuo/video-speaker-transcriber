# Whisper model loading
import torch
import whisper
import matplotlib.pyplot as plt
import numpy as np
import IPython
import IPython.display as ipd
import warnings
import librosa
import copy
warnings.filterwarnings('ignore')

# Extract audio from video files
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
# Language translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# Import face recognition libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import os
import pandas as pd

def load_and_preprocess_audio(file_path, target_sr=16000):
    try:
        """
        Load and preprocess audio file for transcription

        Args:
            file_path (str): Path to audio file
            target_sr (int): Target sample rate in Hz

        Returns:
            tuple: (waveform, sample_rate)
        """
        waveform, original_sr = librosa.load(file_path, sr=None, mono=True)
        print(f"✅ Loaded audio: {file_path}")
        print(f"   Original sample rate: {original_sr} Hz")
        print(f"   Duration: {len(waveform) / original_sr:.2f} seconds")

        # Resample if needed
        if original_sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)
            print(f"   Resampled to {target_sr} Hz")

        # Normalize
        waveform = waveform / np.max(np.abs(waveform))
        print(f"   Audio range: [{waveform.min():.3f}, {waveform.max():.3f}]")

        return waveform, target_sr

    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None, None

def transcribe_with_whisper(model, audio_data, language=None, task="transcribe"):
    """
    Transcribe audio using Whisper model

    Args:
        model: Loaded Whisper model
        audio_data: Audio numpy array
        language: Target language (None for auto-detect)
        task: "transcribe" or "translate"

    Returns:
        result: Whisper transcription result
    """
    # Perform transcription
    result = model.transcribe(
        audio_data,
        language=language,
        task=task,
        verbose=True  # Show progress how the model behaves
    )

    return result

def split_audio_by_segments(segments, audio_array, sample_rate):
    """
    Split audio array into segments based on timestamps

    Args:
        segments: List of segment dictionaries with start/end times
        audio_array: Full audio waveform
        sample_rate: Audio sample rate

    Returns:
        list: List of audio segments
    """
    audio_segments = []

    for seg in segments:
        start_idx = int(seg["start"] * sample_rate)  # 片段开始采样点
        end_idx = int(seg["end"] * sample_rate)      # 片段结束采样点

        seg_audio = audio_array[start_idx:end_idx]  # 截取该片段的波形
        audio_segments.append(seg_audio)

    return audio_segments

def language_overlap_prevention(result, max_duration=5.0):
    """
    Prevent language overlap by merging short segments

    Args:
        result: Whisper transcription result
        max_duration: Maximum duration for short segments

    Returns:
        result: Modified transcription result
    """
    result_new = copy.deepcopy(result)  # 深拷贝
    i = 0
    while i < len(result_new["segments"]) - 1:
        current_seg = result_new["segments"][i]
        if (current_seg["end"] - current_seg["start"] <= max_duration):
            next_seg = result_new["segments"][i + 1]
            next_seg["start"] = current_seg["start"]
            next_seg["text"] = current_seg["text"] + next_seg["text"]
            result_new["segments"].pop(i)  # 删除当前短片段
        else:
            i += 1
    return result_new

# 🔹 2. Detect language for each segment
def language_detection_whisper(result, audio_data, model):
    """
    Detect language for each segment using Whisper

    Args:
        result: Whisper transcription result
        audio_data: Audio numpy array
        model: Loaded Whisper model

    Returns:
        list: Detected languages for each segment
    """
    
    sample_rate = 16000
    segments = result.get("segments", [])
    segment_langs = []
    for seg in segments:
        start = int(seg["start"] * sample_rate)
        end = int(seg["end"] * sample_rate)
        seg_audio = audio_data[start:end]
        if len(seg_audio) == 0:
          seg["detected_language"] = "unknown"
          seg["language_prob"] = 0.0
          continue
        seg_audio = whisper.pad_or_trim(seg_audio)
        mel = whisper.log_mel_spectrogram(torch.tensor(seg_audio).to(model.device))
        _, probs = model.detect_language(mel)

        lang = max(probs, key=probs.get)
        segment_langs.append(lang)

    return segment_langs

# 建立facebook/nllb-200-distilled-600M模型
def setup_translation_models(model_name="facebook/nllb-200-distilled-600M"):
    """
    Set up NLLB translation models for multiple language pairs

    Args:
        model_name: NLLB model variant to use

    Returns:
        Dictionary of translation pipelines
    """

    print(f"🔧 Setting up NLLB Translation Models:")
    print(f"   Model: {model_name}")
    print(f"   Loading tokenizer and model...")

    # Load model and tokenizer都根据这个"facebook/nllb-200-distilled-600M"模型确定
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print(f"   ✅ Model loaded successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            print(f"   📱 Model moved to GPU")

    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None

    # Define common language pairs定义常用语言对
    # src 表示源语言（这里是 NLLB 规定的语言代码）。tgt 表示目标语言。name 用于打印时显示的语言对名称。
    language_pairs = {
        'fr_to_en': {'src': 'fra_Latn', 'tgt': 'eng_Latn', 'name': 'French → English'},
        'es_to_en': {'src': 'spa_Latn', 'tgt': 'eng_Latn', 'name': 'Spanish → English'},
        'de_to_en': {'src': 'deu_Latn', 'tgt': 'eng_Latn', 'name': 'German → English'},
        'zh_to_en': {'src': 'zho_Hans', 'tgt': 'eng_Latn', 'name': 'Chinese → English'},
        'ja_to_en': {'src': 'jpn_Jpan', 'tgt': 'eng_Latn', 'name': 'Japanese → English'},
        'ar_to_en': {'src': 'arb_Arab', 'tgt': 'eng_Latn', 'name': 'Arabic → English'},
        'hi_to_en': {'src': 'hin_Deva', 'tgt': 'eng_Latn', 'name': 'Hindi → English'},
        'ru_to_en': {'src': 'rus_Cyrl', 'tgt': 'eng_Latn', 'name': 'Russian → English'},
        'pt_to_en': {'src': 'por_Latn', 'tgt': 'eng_Latn', 'name': 'Portuguese → English'},
        'id_to_en': {'src': 'ind_Latn', 'tgt': 'eng_Latn', 'name': 'Indonesian → English'},
        'ko_to_en': {'src': 'kor_Hang', 'tgt': 'eng_Latn', 'name': 'Korean → English'},
    }


    # Create translation pipelines 用来存储不同语言对的 pipeline
    translators = {}

    print(f"\n🌍 Creating translation pipelines:")# pair_id 是字典的 key，config 是 value
    for pair_id, config in language_pairs.items():# index和元素
        try: # 遍历语言对字典，依次创建翻译 pipeline
            translator = pipeline(
                'translation',
                model=model,
                tokenizer=tokenizer,
                src_lang=config['src'],#source language
                tgt_lang=config['tgt'],#translate to
                max_length=512,
                device=0 if torch.cuda.is_available() else -1
            )
            # 把这个pipeline存储到字典里面，用原来的转换名称作为索引
            translators[pair_id] = {
                'pipeline': translator,
                'config': config
            }

            print(f"   ✅ {config['name']}")# 存储完成

        except Exception as e:
            print(f"   ❌ Failed to create {config['name']}: {e}")

    print(f"\n🎉 Created {len(translators)} translation pipelines")
    return translators # 字典返回

# 识别模型，输入：建立好的pipeline字典translation_models；whisper_model
def create_complete_speech_translation_pipeline(whisper_model, translation_models):
    """
    Create a complete speech-to-translation pipeline

    Args:
        whisper_model: Loaded Whisper model
        translation_models: Dictionary of NLLB translation pipelines

    Returns:
        Pipeline function
    """
    # 输入音频数据，目标语言代码，源语言
    def translate_speech(audio_data, target_language='eng_Latn', source_language=None):
        """
        Complete speech-to-translation pipeline

        Args:
            audio_data: Input audio numpy array
            target_language: Target language code (NLLB format)
            source_language: Source language hint (Whisper format)

        Returns:
            Complete pipeline results
        """
        #初始化 pipeline 结果字典，输出每个步骤的输出、耗时和错误信息。
        pipeline_results = {
            'steps': [],
            'errors': []
        }

        import time

        print(f"🚀 Speech-to-Translation Pipeline:")
        print(f"   Target Language: {target_language}")
        print(f"   Source Language Hint: {source_language or 'auto-detect'}")
        print("-" * 40)

      # 调用 Whisper 进行语音识别
        try:
            asr_result = whisper_model.transcribe(
                audio_data,
                language=source_language,
                task="transcribe",
                temperature=0.0,
                word_timestamps=True
            )
            transcribed_text = asr_result['text'].strip()# 识别到的文本
            detected_language = asr_result.get('language', 'unknown')


            #打印并保存结果
            print(f"   Detected Language: {detected_language} ")
            print(f"   Transcribed Text: '{transcribed_text}'")

            pipeline_results['steps'].append({
                'step': 'asr',
                'status': 'success',
                'output': transcribed_text,
                'language': detected_language,

            })


        except Exception as e:
            print(f"   ❌ ASR Error: {e}")
            pipeline_results['errors'].append(f"ASR: {e}")
            return pipeline_results

        # Step 2: Text Translation with NLLB
        print(f"\n🌍 Step 2: Text Translation")
        start_time = time.time()

        # Find appropriate translator找寻合适的翻译器——通过目标语言是否相同
        target_translator = None
        for pair_id, translator_info in translation_models.items():
            if translator_info['config']['tgt'] == target_language:
                target_translator = translator_info
                break

        if not target_translator:
            error_msg = f"No translator available for target language: {target_language}"
            print(f"   ❌ {error_msg}")
            pipeline_results['errors'].append(error_msg)
            return pipeline_results

        try:
            # 执行翻译操作
            translation_result = target_translator['pipeline'](transcribed_text) #转换完成的文本导入到翻译模型中
            translated_text = translation_result[0]['translation_text']#将翻译完成的结果赋值给text

            print(f"   Target Language: {target_translator['config']['name']}")
            print(f"   Translated Text: '{translated_text}'")
            # 将翻译结果信息保存
            pipeline_results['steps'].append({
                'step': 'translation',
                'status': 'success',
                'output': translated_text,
                'target_language': target_language,
                'translator': target_translator['config']['name']
            })

        except Exception as e:
            print(f"   ❌ Translation Error: {e}")
            pipeline_results['errors'].append(f"Translation: {e}")
            return pipeline_results

        # Step 3: Results Summary


        print(f"\n📊 Pipeline Summary:")
        print(f"   Final Result: '{translated_text}'")

        pipeline_results['summary'] = {
            'source_text': transcribed_text,
            'target_text': translated_text,
            'source_language': detected_language,
            'target_language': target_language,
            'success': True
        }

        return pipeline_results

    return translate_speech #函数名称就是返回值，返回调用的函数

def segment_time_transcrptions(result, translation_results):
    """
    Map transcription segments to translation results with timestamps

    Args:
        result: Whisper transcription result
        translation_results: List of translated segments

    Returns:
        list: List of [timestamp, text] pairs
    """
    transcriptions = []
    k = 0
    for i in result["segments"]:
        transcriptions.append([i["start"], translation_results[k]])
        k += 1
    return transcriptions

def get_frames_with_time(video_path, interval_seconds=3):
    """
    Extract video frames at regular intervals

    Args:
        video_path: Path to video file
        interval_seconds: Interval in seconds between frames

    Returns:
        tuple: (frames, timestamps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_interval = int(fps * interval_seconds)  # 每隔多少帧提取一次
    timestamps = []
    saved_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            #这里是每隔2秒的帧
            saved_frames.append(frame)
            timestamps.append(frame_idx / fps)
        frame_idx += 1
    cap.release()

    return saved_frames, timestamps

def detect_face(frames, mtcnn):
    """
    Detect faces in video frames

    Args:
        frames: List of video frames
        mtcnn: Face detection model

    Returns:
        list: Detected face regions
    """
    i = 0
    faces_picture = []
    face_regions = []
    while i < len(frames):
        frame_rgb = cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
        aligned_faces, probs = mtcnn(Image.fromarray(frame_rgb), return_prob=True)
        if aligned_faces is not None:
            face_regions.append(aligned_faces)
        else:
            face_regions.append('No Face')
        i += 1
    return face_regions

def embed_face (face_regions, face_recognizer, device):
    """
    Generate face embeddings for recognition

    Args:
        face_regions: List of detected faces
        face_recognizer: Face recognition model
        device: Computing device (CPU/GPU)

    Returns:
        list: Face embeddings
    """
    face_embeddings = []
    for aligned_faces in face_regions:
        # Generate embedding
        face_tensor = aligned_faces.unsqueeze(0).to(device)
        with torch.no_grad():
            new_embedding = face_recognizer(face_tensor).detach().cpu()
            face_embeddings.append(new_embedding)
    return face_embeddings

def build_face_database(face_embeddings):
    """
    Build database of unique faces

    Args:
        face_embeddings: List of face embeddings

    Returns:
        tuple: (face_database, face_clusters)
    """
    face_database = []
    number = []
    k = 2
    for idx, face_embedding in enumerate(face_embeddings):
        if not face_database:
            face_database.append(face_embedding)
            number.append([idx])
        else:
            distances = []
            for db_embedding in face_database:
                distance = torch.norm(face_embedding - db_embedding).item()
                distances.append(distance)

            min_distance = min(distances)
            max_distance = max(distances)
            if min_distance < 0.9 or (max_distance-min_distance)>0.15:
                number[distances.index(min_distance)].append(idx)
            else:
                face_database.append(face_embedding)
                number.append([idx])
            k +=1
    return face_database, number

def speaker_time(number):
    """
    Convert face cluster indices to timestamps

    Args:
        number: List of face cluster indices

    Returns:
        list: Timestamps for each speaker
    """
    time = [[elem * 3 for elem in sublist] for sublist in number]
    return time

def speaker_label(number):
    """
    Generate speaker labels (Speaker_A, Speaker_B, etc.)

    Args:
        number: List of speaker clusters

    Returns:
        list: Speaker labels
    """
    labels = []
    k = 0
    for i in range(len(number)):
        # 将 A、B、C... 转换成字符，用 chr(65) = 'A'
        label = f"Speaker_{chr(65 + k)}"
        labels.append(label)
        k += 1
    return labels

def match_speaker_times_to_segment(segment_start, speaker_time, speaker_labels, window=1.0):
    """
    Match speaker to transcription segment based on timing

    Args:
        segment_start: Segment start time
        speaker_time: List of speaker timestamps
        speaker_labels: List of speaker labels
        window: Time window for matching

    Returns:
        str: Matched speaker label
    """
    for i in range(len(speaker_time)):
        for t in speaker_time[i]:
            if segment_start <= t < segment_start + window:
                return speaker_labels[i]
    return "Unknown"

def append_speaker_matches_to_transcriptions(transcriptions, speaker_time, speaker_labels, window=1.0):
    """
    Add speaker labels to transcription segments

    Args:
        transcriptions: List of transcription segments
        speaker_time: List of speaker timestamps
        speaker_labels: List of speaker labels
        window: Time window for matching

    Returns:
        list: Transcriptions with speaker labels
    """
    updated_transcriptions = []
    for segment in transcriptions:
        start_time = segment[0]
        text = segment[1]
        matched = match_speaker_times_to_segment(start_time, speaker_time, speaker_labels, window)
        updated_transcriptions.append([start_time, text, matched])
    return updated_transcriptions

def format_time(seconds):
    """
    Format seconds into HH:MM:SS format

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

def generate_subtitle_lines_from_transcriptions(augmented_transcriptions):
    """
    Generate subtitle lines from transcriptions

    Args:
        augmented_transcriptions: List of transcription segments with speakers

    Returns:
        list: Formatted subtitle lines
    """
    subtitle_lines = []
    for segment in augmented_transcriptions:
        start_time, text, speaker = segment
        timestamp = format_time(start_time)

        if isinstance(text, str):
            cleaned = text.strip()
            if cleaned:
                subtitle_lines.append(f"{timestamp} {speaker}: {cleaned}")
        elif isinstance(text, dict):
            if 'source_text' in text:
                subtitle_lines.append(f"{timestamp} {speaker}: {text['source_text'].strip()}")
            if 'target_text' in text:
                subtitle_lines.append(f"{timestamp} {speaker}: [TRANSLATED] {text['target_text'].strip()}")
    return subtitle_lines

class VideoTranscriber:

    def __init__(self, whisper_model_size='base'):
        """Initialize models and components"""
        # 1.Load Whisper model
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Loading Whisper model
        self.model = whisper.load_model(whisper_model_size, device=self.device)
        self.translation_models = setup_translation_models()
        # 2.Setup face recognition components
        self.mtcnn = MTCNN(
            image_size=160,        # Output size for detected faces
            margin=0,              # Margin around detected face
            min_face_size=20,      # Minimum face size to detect
            thresholds=[0.6, 0.7, 0.7],  # Detection thresholds for 3 stages
            factor=0.709,          # Scaling factor between levels
            post_process=True,     # Apply post-processing
            device=self.device
        )
        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # 3.Initialize configuration
        # Set random seed for reproducibility
        torch.random.manual_seed(42)
        np.random.seed(42)
    def extract_audio(self, video_path):
        """Extract audio track from video file"""
        # 1.Use moviepy or similar library
        # 读取视频
        clip = VideoFileClip(video_path)
        # 从视频变量中读取音频
        audio = clip.audio
        # 导出音频到相对路径
        file_path = "test_materials/output.wav"
        audio.write_audiofile(file_path)

        # 2.Convert to appropriate format for Whisper
        # 3.Return audio data and metadata
        audio_data, sample_rate =load_and_preprocess_audio(file_path)
        audio_source = "input transferred sample"
        return audio_data
    def extract_faces(self, video_path, interval=3):
        """Extract faces from video at regular intervals"""
        # 1.Use OpenCV to read video frames and get timing information
        frames, timestamps = get_frames_with_time(video_path, interval)
        # 2.Detect faces using face detection
        face_regions = detect_face(frames, self.mtcnn)
        # 3.Extract face regions with timestamps
        face_data = embed_face(face_regions, self.face_recognizer, self.device)
        return face_data

    def build_speaker_database(self, face_data):
        """Create speaker identification database"""
        # 1.Cluster similar faces together
        face_database, number = build_face_database(face_data)
        # 2.Assign speaker labels (Speaker_A, Speaker_B, etc.)
        speaker_labels = speaker_label(number)
        # 3.Create mapping from time ranges to speakers
        speaker_timeline = speaker_time(number)
        return speaker_timeline, speaker_labels

    def transcribe_segments(self, audio_data):
        """Generate transcription with language detection"""
        # 1.Use Whisper for transcription
        self.result = transcribe_with_whisper(self.model, audio_data)
        # 2.Handle overlapping or unclear segments
        self.result = language_overlap_prevention(self.result, max_duration=2.0)
        # 3.Detect language per segment
        audio_segments = split_audio_by_segments(self.result['segments'], audio_data, sample_rate=16000)
        segment_langs = language_detection_whisper(self.result, audio_data, self.model)
        # 4.Generate translations when needed
        # 将这个调用的函数speech_translator等价于translate_speech
        speech_translator = create_complete_speech_translation_pipeline(self.model, self.translation_models)
        # 5.Return structured transcription data with timestamps
        target_lang ='eng_Latn'
        self.translation_results = []
        k = 0
        for seg in segment_langs:

          if seg == 'en':
            self.translation_results.append(self.result["segments"][k]['text'])
          else:
            result_translation = speech_translator(audio_segments[k], target_language=target_lang)# 相当于调用translate_speech
            source = result_translation['summary']['source_text']
            target = result_translation['summary']['target_text']
            # 构建以文本为索引的字典
            text_dict = {
                'source_text': source,
                'target_text': target
            }
            self.translation_results.append(text_dict)
          k += 1

        transcription = segment_time_transcrptions(self.result, self.translation_results)
        return transcription
    def align_speakers_and_speech(self, transcription, speaker_timeline, speaker_labels, interval):
        """Match speech segments with identified speakers"""
        # 1.Temporal alignment of speech and video and assign speaker labels to transcript segments
        aligned_data = append_speaker_matches_to_transcriptions(transcription, speaker_timeline, speaker_labels, interval)
        return aligned_data

    def format_output(self, aligned_data, output_path):
        """Generate final formatted transcript"""
        # 1.Create timestamp formatted output and include translations where applicable
        subtitle_lines = generate_subtitle_lines_from_transcriptions(aligned_data)
        # 2.Write to specified output file
        with open(output_path, "w", encoding="utf-8") as f:
          f.write("\n".join(subtitle_lines))

    def process_video(self, video_path, output_path):
        """Final arrangement of the workline."""
        audio_data = self.extract_audio(video_path)
        transcription = self.transcribe_segments(audio_data)
        interval = 3
        face_data = self.extract_faces(video_path, interval)
        speaker_timeline, speaker_labels = self.build_speaker_database(face_data)
        aligned_data = self.align_speakers_and_speech(transcription, speaker_timeline, speaker_labels, interval)
        self.format_output(aligned_data, output_path)

def audio_transcriber(input_file, output_file):
    """
    Process audio/video file and generate intelligent transcript

    Args:
        input_file (str): Path to input audio/video file
        output_file (str): Path to output transcript (.txt file)

    Returns:
        dict: Processing statistics and metadata (optional)
    """
    transcriber = VideoTranscriber()
    return transcriber.process_video(input_file, output_file)

input_file = "test_materials/input1.mp4"
#input_file = "test_materials/input2.mp4"
output_file = "output1.srt"
#output_file = "output2.srt"
audio_transcriber(input_file, output_file)
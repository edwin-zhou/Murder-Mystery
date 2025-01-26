import base64
import io
import re
from typing import Optional
from google.cloud import speech
from neo4j import GraphDatabase
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import ffmpeg
from openai import OpenAI
import requests
from imageUtils import format_images_to_openai_content

import json

from openai.types.audio import TranscriptionSegment
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Pyannote
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
hf_token = os.getenv("HF_TOKEN")
openai_key = os.getenv("OPENAI_TOKEN")
openai_client = OpenAI(
    api_key=openai_key
)

# Replace these with your actual Neo4j Aura credentials
URI = "neo4j+s://d381ec68.databases.neo4j.io"
AUTH = ("neo4j", "5vlCBP1hnzfqNyqeLbSuu6PQ_1pY6b0n_XDpfLRQPoU")

# Create a driver instance
driver = GraphDatabase.driver(URI, auth=AUTH)
try:
    driver.verify_connectivity()
    print("Connection to Neo4j is good.")
except Exception as e:
    print(f"Connection to Neo4j failed: {e}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# Initialize pyannote Speaker Diarization pipeline
# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
# and 'pyannote/speaker-diarization' with the correct model name or path
# diarization_pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization",
#     use_auth_token=hf_token
# )

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    1. Save uploaded file locally.
    2. Detect scene boundaries.
    3. Extract audio.
    4. Run speaker diarization.
    5. Extract screenshots for each scene.
    6. Return scene info + diarization + screenshot paths.
    """

    # 1. Save the uploaded file
    file_id = str(uuid.uuid4())  # Unique ID for this file
    input_video_path = f"{file_id}"
    with open(input_video_path, "wb") as f:
        f.write(await file.read())

    # 2. Detect scene boundaries using PySceneDetect
    scenes = detect_scenes(input_video_path)

    # 3. Extract audio (entire track for better diarization)
    audio_path = f"{file_id}_audio.wav"
    extract_audio(input_video_path, audio_path)

    # 4. Perform speaker diarization on the extracted audio
    diarization_result = run_diarization(audio_path)

    # 5. Extract screenshots for each scene at 0.2 fps
    #    We'll store them in a directory named after the file_id
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_info = extract_screenshots(input_video_path, scenes, screenshot_dir)

    # Build a response
    response_data = {
        "file_id": file_id,
        "scenes": [],
        "diarization": diarization_result,
    }

    for idx, (start_sec, end_sec) in enumerate(scenes):
        scene_item = {
            "scene_index": idx,
            "start": start_sec,
            "end": end_sec,
            "screenshots": screenshot_info[idx]  # list of file paths
        }
        response_data["scenes"].append(scene_item)

    scene_data = aggregate_scene_data(response_data)

    # 6. Build prompts for each scene and process them
    llm_response_queries = process_scenes(scene_data)

    # 7. Run the generated queries against Neo4j
    db_results = []
    for query in llm_response_queries:
        db_results.append(run_query(query))

    return JSONResponse(content=db_results)


def detect_scenes(video_path: str, threshold: float = 30.0):
    """
    Uses PySceneDetect to detect scenes (abrupt cuts) with a given threshold.
    Returns a list of (start_time_sec, end_time_sec).
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()

    # Convert from FrameTimecodes to (start_sec, end_sec)
    scenes = []
    for scene in scene_list:
        start = scene[0].get_seconds()  # float
        end = scene[1].get_seconds()  # float
        scenes.append((start, end))

    # If no scenes found, return the entire duration as a single "scene"
    if not scenes:
        duration = video_manager.get_base_timecode().frames_to_seconds(video_manager.get_frame_count())
        scenes = [(0.0, duration)]

    video_manager.release()
    return scenes


def extract_audio(video_path: str, audio_path: str):
    """
    Extracts audio from the video using ffmpeg-python and saves as WAV (16-bit PCM).
    """
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format='wav', acodec='pcm_s16le', ac=1)  # mono track
        .overwrite_output()
        .run(quiet=True)
    )


def transcribe_audio(audio_path: str) -> Optional[list[TranscriptionSegment]]:
    """
    Transcribe audio using OpenAI's Whisper API.

    Args:
        audio_path (str): Path to the audio file.
        api_key (str): OpenAI API key.
        model (str): Whisper model to use (default: "whisper-1").

    Returns:
        list: List of transcription segments with {start, end, text}.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"

    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        return transcript.segments


def run_diarization(audio_path: str) -> list:
    """
    Run speaker diarization and align with transcription using Google Cloud Speech-to-Text API.
    Returns a list of {start, end, speaker, text} segments.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        list: Aligned results with {start, end, speaker, text}.
    """
    # Initialize the Speech-to-Text client
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\eyyypc\PycharmProjects\Murder-Mystery\backend\ensemble-demo-421315-ab61fb41873d.json"

    client = speech.SpeechClient()

    # Read the audio file
    with io.open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=10,
    )

    # Configure the request
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        diarization_config=diarization_config,
        enable_word_time_offsets=True,
    )

    # Load audio data
    audio = speech.RecognitionAudio(content=audio_content)

    # Perform transcription with diarization
    response = client.recognize(config=config, audio=audio)

    # Process results into segments
    aligned_results = []
    current_segment = {
        "start": None,
        "end": None,
        "speaker": None,
        "text": ""
    }

    for result in response.results:
        # The first alternative is typically the most accurate
        alternative = result.alternatives[0]

        for word_info in alternative.words:
            word_start = word_info.start_time.total_seconds()
            word_end = word_info.end_time.total_seconds()
            speaker_tag = word_info.speaker_tag  # Speaker ID
            word_text = word_info.word

            # Check if the speaker changed or it's the first word
            if current_segment["speaker"] != f"Speaker {speaker_tag}":
                # If there is an existing segment, finalize it
                if current_segment["speaker"]:
                    aligned_results.append(current_segment)

                # Start a new segment
                current_segment = {
                    "start": word_start,
                    "end": word_end,
                    "speaker": f"Speaker {speaker_tag}",
                    "text": word_text
                }
            else:
                # Continue the current segment
                current_segment["end"] = word_end
                current_segment["text"] += f" {word_text}"

    # Append the last segment
    if current_segment["speaker"]:
        aligned_results.append(current_segment)

    return aligned_results


def extract_screenshots(video_path: str, scenes: list, output_dir: str):
    """
    For each scene, extract frames at 0.2 fps (1 frame every 5 seconds).
    Return a list (indexed by scene) of lists of image paths.
    """
    screenshot_data = []
    for idx, (start_sec, end_sec) in enumerate(scenes):
        scene_duration = end_sec - start_sec
        if scene_duration <= 0:
            # Edge case: skip
            screenshot_data.append([])
            continue

        output_pattern = os.path.join(output_dir, f"scene_{idx}_%03d.jpg")

        # Using ffmpeg-python to extract frames at 1/5 fps
        fps = 1 / 5 if scene_duration >= 5 else 1 / scene_duration  # Adjust fps for short scenes
        try:
            (
                ffmpeg
                .input(video_path, ss=start_sec, t=scene_duration)
                .filter('fps', fps=fps)
                .output(output_pattern)
                .overwrite_output()
                .run()
            )
        except Exception as e:
            print(f"Error extracting screenshots for scene {idx}: {e}")
            continue
        # Collect the generated files
        # They follow pattern scene_{idx}_001.jpg, scene_{idx}_002.jpg, etc.
        # We'll list them in sorted order
        scene_images = []
        for root, dirs, files in os.walk(output_dir):
            for file_name in sorted(files):
                # match pattern: scene_{idx}_NNN.jpg
                if file_name.startswith(f"scene_{idx}_"):
                    scene_images.append(os.path.join(root, file_name))

        screenshot_data.append(scene_images)

    return screenshot_data


def aggregate_scene_data(data) -> list[dict]:
    """
    data is expected to have two keys:
        - "scenes": a list of scene dicts, each with:
            scene_index, start, end, screenshots
        - "diarization": a list of transcript segments, each with:
            start, end, speaker, text

    This function will return a list of scene dictionaries, where each
    includes all diarization segments that occur within that scene's time range.
    """
    all_scenes = data["scenes"]
    all_segments = data["diarization"]

    aggregated_scenes = []

    for scene in all_scenes:
        scene_index = scene["scene_index"]
        scene_start = scene["start"]
        scene_end = scene["end"]
        screenshots = scene.get("screenshots", [])

        # Collect diarization segments that fall into [scene_start, scene_end]
        relevant_segments = []
        for seg in all_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            # Check if the segment is within the scene's time window
            if seg_start >= scene_start and seg_end <= scene_end:
                relevant_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": seg["speaker"],
                    "text": seg["text"]
                })

        aggregated_scenes.append({
            "scene_index": scene_index,
            "start": scene_start,
            "end": scene_end,
            "screenshots": screenshots,
            "segments": relevant_segments
        })

    return aggregated_scenes


def build_prompt_for_scene(scene):
    """
    scene: dict with keys:
      - scene_index
      - start
      - end
      - screenshots (list)
      - segments (list of diarization segments)
    """
    # Combine the transcript text from all segments
    transcript_text = ""
    for seg in scene["segments"]:
        start_time = seg["start"]
        end_time = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"]
        transcript_text += f"[{start_time}-{end_time}] {speaker}: {text}\n"

    # Example instructions telling the LLM what we want:
    instructions = f"""
        You are an AI specialized in extracting scene-level data for a detective knowledge graph.
        We have the following scene (index = {scene["scene_index"]}):
        
        Scene start: {scene["start"]} seconds
        Scene end: {scene["end"]} seconds

        Transcript segments:
        {transcript_text}
        
        Task:
        1. Identify any key events, relationships, or evidence mentioned in this scene.
        2. Return a Cypher query (for Neo4j) that would insert or update the relevant nodes and relationships in our graph.
           - We have a schema with (Person, Event, Evidence, etc.). 
           - Use MERGE or CREATE as necessary.
           - Add relevant properties from the scene, including times, speaker references, etc.
        
        Output ONLY the Cypher query as text, nothing else.
        """

    images = [format_images_to_openai_content(screenshot) for screenshot in scene['screenshots']]

    return [{"type": "text", "text": instructions}] + images


def process_scenes(scene_transcripts):
    """
    For each scene, build the prompt, attach each screenshot as a Base64 data URL,
    and send the request to OpenAI ChatCompletion.
    """
    queries = []
    for scene_info in scene_transcripts:
        # 1. Build your textual prompt (scene metadata + transcript)
        prompt = build_prompt_for_scene(scene_info)

        # 3. Combine the textual prompt plus the image data into the message array
        #    There are a few ways to structure these. Below, we place the text prompt
        #    in one user message, then place each image object in its own user message
        #    in JSON form. Adjust this as needed for your use case.

        # 4. Call the OpenAI ChatCompletion endpoint
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        # Extract the LLM response
        query = response.choices[0].message.content

        print("----- Scene Index:", scene_info["scene_index"], "-----")
        print(query)
        print("---------------------------------------------\n")

        # Optionally: execute the query against Neo4j
        # execute_in_neo4j(query)
        queries.append(query)
    return queries


def run_query(raw_query):
    # Remove all occurrences of ``` and escape sequences
    cleaned_query = re.sub(r'```(?:cypher)?\n', '', raw_query)  # Removes starting ```cypher\n
    cleaned_query = re.sub(r'\\n```$', '', cleaned_query)       # Removes ending \n```
    cleaned_query = re.sub(r'```', '', cleaned_query)           # Removes stray ```
    # Replace escaped newlines with actual newlines
    cleaned_query = cleaned_query.replace('\\n', '\n')
    try:
        with driver.session() as session:
            result = session.run(cleaned_query.strip())
            return {"success": True, "data": result.data()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# # Generate the OpenAPI schema
# openapi_schema = app.openapi()
#
# # Save the schema to a JSON file
# with open("openapi_schema.json", "w") as f:
#     json.dump(openapi_schema, f)


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()

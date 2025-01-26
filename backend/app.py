from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid
import ffmpeg
import json

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

app = FastAPI()

# Initialize pyannote Speaker Diarization pipeline
# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
# and 'pyannote/speaker-diarization' with the correct model name or path
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=hf_token
)

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
    diarization_result = run_diarization(audio_path, diarization_pipeline)

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

    return JSONResponse(content=response_data)

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
        end = scene[1].get_seconds()    # float
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

def run_diarization(audio_path: str, pipeline):
    """
    Run speaker diarization using pyannote.audio pipeline.
    Returns a list of { start, end, speaker } segments.
    """
    # Inference
    diarization = pipeline(audio_path)

    # Build a structured result
    diarization_result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_result.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return diarization_result

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

# Generate the OpenAPI schema
openapi_schema = app.openapi()

# Save the schema to a JSON file
with open("openapi_schema.json", "w") as f:
    json.dump(openapi_schema, f)


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
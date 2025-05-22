import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import base64
import sys

# Add project root to sys.path if tests are run from tests/ directory
# This ensures 'main' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, ModelProcessor

TEST_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "test_samples")
SAMPLE_VIDEO_PATH = os.path.join(TEST_SAMPLES_DIR, "sample_video.mp4")
INVALID_FILE_PATH = os.path.join(TEST_SAMPLES_DIR, "invalid_file.txt")

@pytest.fixture(scope="session", autouse=True)
def manage_test_files():
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)

    # Check if sample_video.mp4 was created by the bash command in the previous step
    # If not, this fixture won't attempt to recreate it here as per instructions.
    # The ffmpeg command should have already run.

    with open(INVALID_FILE_PATH, "w") as f:
        f.write("This is not a video.")
    
    yield 

    # Clean up sample files after test session
    # sample_video.mp4 is managed by the run_in_bash_session call, so only remove invalid_file.txt
    if os.path.exists(INVALID_FILE_PATH):
        os.remove(INVALID_FILE_PATH)
    
    # Attempt to remove TEST_SAMPLES_DIR only if it's empty AND it's not the one created by ffmpeg directly
    # For simplicity, and given the previous step created the video, we'll leave the dir
    # if it contains the video. If it's empty after removing invalid_file.txt, try to remove.
    if os.path.exists(TEST_SAMPLES_DIR) and not os.listdir(TEST_SAMPLES_DIR):
        try:
            os.rmdir(TEST_SAMPLES_DIR)
        except OSError:
            pass


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.mark.asyncio
async def test_summarize_video_with_gemini_success():
    mock_gemini_response_obj = MagicMock()
    mock_gemini_response_obj.text = "This is a mocked summary."
    
    # Mock the return value of the inner function run_gemini_summarization_api
    # which is called by run_in_thread
    async def mock_run_in_thread_function(func, *args, **kwargs):
        # Directly return the desired mock response, simulating the behavior 
        # of the `run_gemini_summarization_api` when called within `run_in_thread`.
        return mock_gemini_response_obj.text 

    with patch('main.run_in_thread', new=mock_run_in_thread_function):
        dummy_frames = ["base64image1", "base64image2"]
        prompt = "Summarize these frames."
        # The actual ModelProcessor.summarize_video_with_gemini returns a dict
        result_dict = await ModelProcessor.summarize_video_with_gemini(dummy_frames, prompt)
        assert result_dict["status"] == "success"
        assert result_dict["summary"] == "This is a mocked summary."
        assert result_dict["model"] == "gemini"

@pytest.mark.asyncio
async def test_summarize_video_with_gemini_api_error():
    async def mock_run_in_thread_raises_exception(func, *args, **kwargs):
        raise Exception("Gemini API Error")

    with patch('main.run_in_thread', new=mock_run_in_thread_raises_exception):
        dummy_frames = ["base64image1"]
        prompt = "Summarize."
        result_dict = await ModelProcessor.summarize_video_with_gemini(dummy_frames, prompt)
        assert result_dict["status"] == "error"
        assert "Gemini API error during summarization: Gemini API Error" in result_dict["message"]
        assert result_dict["model"] == "gemini"


def test_summarize_api_success(client):
    if not os.path.exists(SAMPLE_VIDEO_PATH) or os.path.getsize(SAMPLE_VIDEO_PATH) < 100: 
        pytest.skip("Valid sample video not available or too small. FFMPEG step might have failed or produced an invalid file.")

    # Mock the ModelProcessor.summarize_video_with_gemini method
    mock_summary_response = {
        "status": "success",
        "summary": "Mocked video summary from API test.",
        "model": "gemini", # Ensure this matches what ModelProcessor would return
        "timestamp": 12345
    }
    with patch('main.ModelProcessor.summarize_video_with_gemini', return_value=mock_summary_response) as mock_summarize:
        response = client.post("/api/video/summarize", json={"video_path": SAMPLE_VIDEO_PATH})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["summary"] == "Mocked video summary from API test."
        assert data["model_used"] == "gemini" 
        assert "frames_processed" in data
        # The sample video is 3 seconds long, default interval is 5s.
        # Frame extraction logic: 0s. So 1 frame expected.
        assert data["frames_processed"] >= 1, f"Expected at least 1 frame, got {data['frames_processed']}"
        mock_summarize.assert_called_once()


def test_summarize_api_video_not_found(client):
    response = client.post("/api/video/summarize", json={"video_path": "non_existent_video.mp4"})
    assert response.status_code == 404 
    data = response.json()
    assert data["status"] == "error"
    assert "Video file not found" in data["message"]


def test_summarize_api_invalid_video_file(client):
    response = client.post("/api/video/summarize", json={"video_path": INVALID_FILE_PATH})
    # Expecting 500 because cv2.VideoCapture likely fails and might not be caught as a specific client error.
    # Or 400 if specific checks for video properties (like duration 0) are hit first.
    assert response.status_code in [400, 500] 
    data = response.json()
    assert data["status"] == "error"
    # Check for messages that might come from cv2 errors or duration checks
    possible_messages = ["Could not open video file", "Video duration is zero or FPS is invalid"]
    assert any(msg in data["message"] for msg in possible_messages)


def test_summarize_api_gemini_call_error(client):
    if not os.path.exists(SAMPLE_VIDEO_PATH) or os.path.getsize(SAMPLE_VIDEO_PATH) < 100:
        pytest.skip("Valid sample video not available or too small. FFMPEG step might have failed.")

    # Mock ModelProcessor.summarize_video_with_gemini to return an error structure
    mock_error_response = {
        "status": "error",
        "message": "Simulated Gemini Error from ModelProcessor",
        "model": "gemini"
    }
    with patch('main.ModelProcessor.summarize_video_with_gemini', return_value=mock_error_response) as mock_summarize_error:
        response = client.post("/api/video/summarize", json={"video_path": SAMPLE_VIDEO_PATH})
        # The endpoint itself should still succeed in calling the (mocked) ModelProcessor
        # The error is from the *logic* of summarization, not an unhandled API exception in this case
        assert response.status_code == 500 # The endpoint should return 500 if the summarization failed
        data = response.json()
        assert data["status"] == "error"
        assert data["message"] == "Simulated Gemini Error from ModelProcessor"
        mock_summarize_error.assert_called_once()

def test_summarize_api_missing_video_path(client):
    response = client.post("/api/video/summarize", json={}) # Missing video_path
    assert response.status_code == 422 # FastAPI's unprocessable entity for missing fields
    data = response.json()
    assert "detail" in data
    assert any(d["msg"] == "Field required" and d["loc"] == ["body", "video_path"] for d in data["detail"])

def test_summarize_api_empty_video_path(client):
    response = client.post("/api/video/summarize", json={"video_path": ""})
    # This will likely be caught by the os.path.exists check.
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert "Video file not found" in data["message"]

# Example test for frame extraction parameters
def test_summarize_api_custom_extraction_params(client):
    if not os.path.exists(SAMPLE_VIDEO_PATH) or os.path.getsize(SAMPLE_VIDEO_PATH) < 100:
        pytest.skip("Valid sample video not available. FFMPEG step might have failed.")

    mock_summary_response = {
        "status": "success",
        "summary": "Custom params summary.",
        "model": "gemini",
        "timestamp": 12345
    }
    with patch('main.ModelProcessor.summarize_video_with_gemini', return_value=mock_summary_response) as mock_summarize:
        # Sample video is 3s. Interval 1s should give 3 frames (0s, 1s, 2s). Max frames 2.
        # Frame indices: [0, 10, 20] (for 10fps video)
        # Selected due to max_frames=2: [0, 20] -> 2 frames
        response = client.post("/api/video/summarize", json={
            "video_path": SAMPLE_VIDEO_PATH,
            "extraction_interval_seconds": 1,
            "max_frames": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["summary"] == "Custom params summary."
        assert data["frames_processed"] == 2 # Expect 2 frames due to max_frames
        mock_summarize.assert_called_once()

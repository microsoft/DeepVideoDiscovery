import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = "360" # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
CLIP_SECS = 10 # seconds

# ------------------ model configuration ------------------ #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None) # will overwrite Azure OpenAI setting

AOAI_CAPTION_VLM_ENDPOINT_LIST = [""]
AOAI_CAPTION_VLM_MODEL_NAME = "gpt-4.1-mini"

AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [""]
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "o3"

AOAI_TOOL_VLM_ENDPOINT_LIST = [""]
AOAI_TOOL_VLM_MODEL_NAME = "gpt-4.1-mini"
AOAI_TOOL_VLM_MAX_FRAME_NUM = 50

AOAI_EMBEDDING_RESOURCE_LIST = [""]
AOAI_EMBEDDING_LARGE_MODEL_NAME = "text-embedding-3-large"
AOAI_EMBEDDING_LARGE_DIM = 3072

# ------------------ agent and tool setting ------------------ #
LITE_MODE = True # if True, only leverage srt subtitle, no pixel downloaded or pixel captioning
GLOBAL_BROWSE_TOPK = 300
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 3  # Maximum number of iterations for the agent to run
import os
from pathlib import Path
import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

app_root_directory = str(Path(__file__).parent.absolute().parents[0])
app_data_dir = os.path.join(app_root_directory, "tempfiles")
output_project_dir = os.path.join(app_data_dir, "output_project_dir")
static_dir = Path(os.path.join(app_data_dir, "preview_files"))

if sly.is_production():
    app_session_id = sly.io.env.task_id()
else:
    app_session_id = 777

local_det_preview_path = os.path.join(static_dir, "det_labeled.jpg")
remote_det_preview_path = os.path.join(
    sly.output.RECOMMENDED_EXPORT_PATH,
    sly.app.fastapi.get_name_from_env(),
    str(app_session_id),
    "det_preview.jpg",
)

local_pose_preview_path = os.path.join(static_dir, "pose_labeled.jpg")
remote_pose_preview_path = os.path.join(
    sly.output.RECOMMENDED_EXPORT_PATH,
    sly.app.fastapi.get_name_from_env(),
    str(app_session_id),
    "pose_preview.jpg",
)

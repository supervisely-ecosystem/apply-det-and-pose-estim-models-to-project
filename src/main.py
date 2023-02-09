import os
from pathlib import Path
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import yaml
from supervisely.app.widgets import (
    Container,
    Card,
    SelectAppSession,
    InputNumber,
    Input,
    Button,
    Field,
    Progress,
    SelectDataset,
    Image,
    ModelInfo,
    ClassesTable,
    DoneLabel,
    ProjectThumbnail,
    Text,
    Editor,
)


# function for updating global variables
def update_globals(new_dataset_ids):
    global dataset_ids, project_id, workspace_id, team_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids:
        project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        team_id = api.workspace.get_info_by_id(workspace_id).team_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        print(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        team_id = api.workspace.get_info_by_id(workspace_id).team_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    else:
        print("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, team_id, project_info, project_meta = [None] * 5


# authentication
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)

# define global variables
app_root_directory = str(Path(__file__).parent.absolute().parents[0])
sly.logger.info(f"App root directory: {app_root_directory}")
app_data_dir = os.path.join(app_root_directory, "tempfiles")
output_project_dir = os.path.join(app_data_dir, "output_project_dir")
static_dir = Path(os.path.join(app_data_dir, "preview_files"))
os.makedirs(static_dir, exist_ok=True)
sly.fs.clean_dir(static_dir)
det_model_data = {}
pose_model_data = {}


### 1. Dataset selection
dataset_selector = SelectDataset(project_id=project_id, multiselect=True)
card_project_settings = Card(title="Dataset selection", content=dataset_selector)


### 2. Connect to detection model
select_det_model = SelectAppSession(allowed_session_tags=["deployed_nn"])
connect_det_model_button = Button(
    text='<i style="margin-right: 5px" class="zmdi zmdi-power"></i>connect to detection model',
    button_type="success",
    button_size="small",
)
connect_det_model_done = DoneLabel("Detection model connected")
connect_det_model_done.hide()
det_model_stats = ModelInfo()
change_det_model_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>change detection model',
    button_type="warning",
    button_size="small",
    plain=True,
)
change_det_model_button.hide()
connect_det_model_content = Container(
    [
        select_det_model,
        connect_det_model_button,
        connect_det_model_done,
        det_model_stats,
        change_det_model_button,
    ]
)
card_connect_det_model = Card(
    title="Connect to Detection Model",
    description="Select served detection model from list below",
    content=connect_det_model_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_connect_det_model.collapse()
card_connect_det_model.lock()


### 3. Detection model classes
det_classes_table = ClassesTable()
select_det_classes_button = Button("select classes")
select_det_classes_button.hide()
select_other_det_classes_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>select other classes',
    button_type="warning",
    button_size="small",
    plain=True,
)
select_other_det_classes_button.hide()
det_classes_done = DoneLabel()
det_classes_done.hide()
det_model_classes_content = Container(
    [
        det_classes_table,
        select_det_classes_button,
        select_other_det_classes_button,
        det_classes_done,
    ]
)
card_det_model_classes = Card(
    title="Detection Model Classes",
    description="Choose classes that will be kept after prediction, other classes will be ignored",
    content=det_model_classes_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_det_model_classes.collapse()
card_det_model_classes.lock()


### 4.1 Detection settings
det_settings_editor = Editor(language_mode="yaml")
save_det_settings_button = Button("save detection settings")
reselect_det_settings_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>reselect detection settings',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_det_settings_button.hide()
det_settings_done = DoneLabel("Detection settings saved")
det_settings_done.hide()
det_preview_loading = Text("Loading detection inference preview...")
det_preview_loading.hide()
det_settings_content = Container(
    [
        det_settings_editor,
        save_det_settings_button,
        reselect_det_settings_button,
        det_settings_done,
        det_preview_loading,
    ]
)
card_det_settings = Card(
    title="Detection Settings",
    content=det_settings_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)


### 4.2 Detection inference preview
det_line_thickness = InputNumber(value=7, min=1, max=14)
det_line_thickness_f = Field(det_line_thickness, "Line thickness")
det_redraw_image_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>redraw',
    button_type="warning",
    button_size="small",
    plain=True,
)
det_redraw_loading = Text("Redrawing detection inference preview...")
det_redraw_loading.hide()
det_labeled_image = Image()
det_image_preview_content = Container(
    [
        det_line_thickness_f,
        det_redraw_image_button,
        det_redraw_loading,
        det_labeled_image,
    ]
)
card_det_image_preview = Card(
    title="Detection inference preview",
    content=det_image_preview_content,
    collapsable=True,
    lock_message="Choose detection settings to unlock",
)
card_det_image_preview.collapse()
card_det_image_preview.lock()
det_settings_preview_content = Container(
    widgets=[card_det_settings, card_det_image_preview],
    direction="horizontal",
    fractions=[1, 2],
)
card_det_settings_preview = Card(
    title="Detection settings and preview",
    content=det_settings_preview_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_det_settings_preview.collapse()
card_det_settings_preview.lock()


### 5. Connect to pose estimation model
select_pose_model = SelectAppSession(allowed_session_tags=["deployed_nn_keypoints"])
connect_pose_model_button = Button(
    text='<i style="margin-right: 5px" class="zmdi zmdi-power"></i>connect to pose estimation model',
    button_type="success",
    button_size="small",
)
connect_pose_model_done = DoneLabel("Pose estimation model connected")
connect_pose_model_done.hide()
pose_model_stats = ModelInfo()
change_pose_model_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>change pose estimation model',
    button_type="warning",
    button_size="small",
    plain=True,
)
change_pose_model_button.hide()
connect_pose_model_content = Container(
    [
        select_pose_model,
        connect_pose_model_button,
        connect_pose_model_done,
        pose_model_stats,
        change_pose_model_button,
    ]
)
card_connect_pose_model = Card(
    title="Connect to Pose Estimation Model",
    description="Select served pose estimation model from list below",
    content=connect_pose_model_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_connect_pose_model.collapse()
card_connect_pose_model.lock()


### 6.1 Pose estimation settings
pose_settings_editor = Editor(language_mode="yaml")
pose_classes_table = ClassesTable()
pose_classes_table_f = Field(
    pose_classes_table, title="Apply Pose Estimation Model to", description="You can label specific classes of images"
)
save_pose_settings_button = Button("save pose estimation settings")
save_pose_settings_button.hide()
reselect_pose_settings_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>reselect pose estimation settings',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_pose_settings_button.hide()
pose_settings_done = DoneLabel("Pose estimation settings saved")
pose_settings_done.hide()
pose_preview_loading = Text("Loading pose estimation inference preview...")
pose_preview_loading.hide()
pose_settings_content = Container(
    [
        pose_settings_editor,
        pose_classes_table_f,
        save_pose_settings_button,
        reselect_pose_settings_button,
        pose_settings_done,
        pose_preview_loading,
    ]
)
card_pose_settings = Card(
    title="Pose Estimation Settings",
    content=pose_settings_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
### 6.2 Pose estimation inference preview
pose_line_thickness = InputNumber(value=7, min=1, max=14)
pose_line_thickness_f = Field(pose_line_thickness, "Line thickness")
pose_redraw_image_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>redraw',
    button_type="warning",
    button_size="small",
    plain=True,
)
pose_redraw_loading = Text("Redrawing pose estimation inference preview...")
pose_redraw_loading.hide()
pose_labeled_image = Image()
pose_image_preview_content = Container(
    [
        pose_line_thickness_f,
        pose_redraw_image_button,
        pose_redraw_loading,
        pose_labeled_image,
    ]
)
card_pose_image_preview = Card(
    title="Pose estimation inference preview",
    content=pose_image_preview_content,
    collapsable=True,
    lock_message="Choose pose estimation settings to unlock",
)
card_pose_image_preview.collapse()
card_pose_image_preview.lock()
pose_settings_preview_content = Container(
    widgets=[card_pose_settings, card_pose_image_preview],
    direction="horizontal",
    fractions=[1, 2],
)
card_pose_settings_preview = Card(
    title="Pose estimation settings and preview",
    content=pose_settings_preview_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_pose_settings_preview.collapse()
card_pose_settings_preview.lock()


### 7. Output project
output_project_name_input = Input(value="Labeled project")
output_project_name_input_f = Field(output_project_name_input, "Output project name")
apply_models_to_project_button = Button("APPLY MODELS TO PROJECT")
progress_bar = Progress()
output_project_thmb = ProjectThumbnail()
output_project_thmb.hide()
output_project_done = DoneLabel("done")
output_project_done.hide()
output_project_content = Container(
    [
        output_project_name_input_f,
        apply_models_to_project_button,
        progress_bar,
        output_project_thmb,
        output_project_done,
    ]
)
card_output_project = Card(
    title="Output Project",
    description="Start labeling by detection and pose estimation models",
    content=output_project_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_output_project.collapse()
card_output_project.lock()


app = sly.Application(
    layout=Container(
        widgets=[
            card_project_settings,
            card_connect_det_model,
            card_det_model_classes,
            card_det_settings_preview,
            card_connect_pose_model,
            card_pose_settings_preview,
            card_output_project,
        ]
    ),
    static_dir=static_dir,
)


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    update_globals(new_dataset_ids)
    if project_info is not None:
        output_project_name_input.set_value(value=project_info.name + " Labeled")
    card_connect_det_model.unlock()
    card_connect_det_model.uncollapse()


@connect_det_model_button.click
def connect_to_det_model():
    det_session_id = select_det_model.get_selected_id()
    if det_session_id is not None:
        connect_det_model_button.hide()
        connect_det_model_done.show()
        select_det_model.disable()
        det_model_stats.set_session_id(session_id=det_session_id)
        det_model_stats.show()
        change_det_model_button.show()
        det_model_meta_json = api.task.send_request(
            det_session_id,
            "get_output_classes_and_tags",
            data={},
        )
        sly.logger.info(f"Detection model meta: {str(det_model_meta_json)}")
        det_model_data["det_model_meta"] = sly.ProjectMeta.from_json(det_model_meta_json)
        det_model_data["det_session_id"] = det_session_id
        det_classes_table.read_meta(det_model_data["det_model_meta"])
        card_det_model_classes.unlock()
        card_det_model_classes.uncollapse()


@change_det_model_button.click
def change_det_model():
    select_det_model.enable()
    connect_det_model_done.hide()
    det_model_stats.hide()
    change_det_model_button.hide()
    connect_det_model_button.show()
    card_det_model_classes.lock()
    card_det_model_classes.collapse()


@det_classes_table.value_changed
def on_det_classes_selected(selected_det_classes):
    n_det_classes = len(selected_det_classes)
    if n_det_classes > 1:
        select_det_classes_button.text = f"Select {n_det_classes} classes"
    else:
        select_det_classes_button.text = f"Select {n_det_classes} class"
    select_det_classes_button.show()


@select_det_classes_button.click
def select_det_classes():
    det_classes_table.disable()
    det_model_data["det_model_classes"] = det_classes_table.get_selected_classes()
    sly.logger.info(f"Detection model classes: {str(det_model_data['det_model_classes'])}")
    n_det_classes = len(det_model_data["det_model_classes"])
    select_det_classes_button.hide()
    if n_det_classes > 1:
        det_classes_done.text = f"{n_det_classes} classes were selected successfully"
    else:
        det_classes_done.text = f"{n_det_classes} class was selected successfully"
    det_classes_done.show()
    select_other_det_classes_button.show()
    # delete unselected classes from model meta
    det_classes_collection = [cls["title"] for cls in det_model_data["det_model_meta"].to_json()["classes"]]
    det_classes_to_delete = [cls for cls in det_classes_collection if cls not in det_model_data["det_model_classes"]]
    det_model_data["det_model_meta"] = det_model_data["det_model_meta"].delete_obj_classes(det_classes_to_delete)
    sly.logger.info(f"Updated detection model meta: {str(det_model_data['det_model_meta'].to_json())}")
    # get detection custom inference settings
    det_inference_settings = api.task.send_request(
        det_model_data["det_session_id"],
        "get_custom_inference_settings",
        data={},
    )
    if det_inference_settings["settings"] is None or len(det_inference_settings["settings"]) == 0:
        det_inference_settings["settings"] = ""
    elif isinstance(det_inference_settings["settings"], dict):
        det_inference_settings["settings"] = yaml.dump(det_inference_settings["settings"], allow_unicode=True)
    det_settings_editor.set_text(det_inference_settings["settings"])
    card_det_settings_preview.unlock()
    card_det_settings_preview.uncollapse()


@select_other_det_classes_button.click
def select_other_det_classes():
    det_classes_table.enable()
    det_classes_table.clear_selection()
    select_other_det_classes_button.hide()
    det_classes_done.hide()
    card_det_settings_preview.lock()
    card_det_settings_preview.collapse()


@save_det_settings_button.click
def save_det_settings():
    det_settings_editor.readonly = True
    # get detection inference settings
    det_inference_settings = det_settings_editor.get_text()
    det_inference_settings = yaml.safe_load(det_inference_settings)
    det_model_data["det_inference_settings"] = det_inference_settings
    if det_inference_settings is None:
        det_inference_settings = {}
        sly.logger.info("Detection model doesn't support custom inference settings.")
    else:
        sly.logger.info("Detection inference settings:")
        sly.logger.info(str(det_inference_settings))
    save_det_settings_button.hide()
    det_settings_done.show()
    det_preview_loading.show()
    # download input project to ouput project directory
    if os.path.exists(output_project_dir):
        sly.fs.clean_dir(output_project_dir)
    sly.download_project(
        api=api,
        project_id=project_id,
        dest_dir=output_project_dir,
        dataset_ids=dataset_ids,
        save_image_info=True,
        save_images=True,
    )
    global output_project, preview_project, preview_image_info, preview_bboxes, preview_image, preview_det_ann, det_preview_output_path
    output_project = sly.Project(output_project_dir, mode=sly.OpenMode.READ)
    # create project for storing preview images
    if os.path.exists(static_dir):
        sly.fs.clean_dir(static_dir)
    preview_project = output_project.copy_data(static_dir)
    # merge preview project meta with det model meta
    meta_with_det = preview_project.meta.merge(det_model_data["det_model_meta"])
    preview_project.set_meta(meta_with_det)
    # get preview image info
    preview_dataset_info = api.dataset.get_info_by_id(dataset_ids[0])
    preview_image_info = api.image.get_list(preview_dataset_info.id)[0]
    # get image annotation
    preview_det_predictions = api.task.send_request(
        det_model_data["det_session_id"],
        "inference_image_id",
        data={"image_id": preview_image_info.id, "settings": det_inference_settings},
        timeout=500,
    )
    preview_det_ann = preview_det_predictions["annotation"]
    preview_det_ann_objects = preview_det_ann["objects"].copy()
    preview_bboxes = []
    for object in preview_det_ann_objects:
        if object["classTitle"] not in det_model_data["det_model_classes"]:
            preview_det_ann["objects"].remove(object)
        else:
            coordinates = object["points"]["exterior"]
            det_bbox = {"bbox": [coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], 1.0]}
            preview_bboxes.append(det_bbox)
    preview_det_ann = sly.Annotation.from_json(preview_det_ann, preview_project.meta)
    det_preview_output_path = os.path.join(static_dir, "det_labeled.jpg")
    preview_image = api.image.download_np(preview_image_info.id)
    preview_det_ann.draw_pretty(
        bitmap=preview_image.copy(),
        output_path=det_preview_output_path,
        thickness=det_line_thickness.get_value(),
        fill_rectangles=False,
    )
    det_labeled_image.set(url="/static/det_labeled.jpg")
    # show preview
    det_preview_loading.hide()
    card_det_image_preview.uncollapse()
    card_det_image_preview.unlock()
    reselect_det_settings_button.show()
    card_connect_pose_model.unlock()
    card_connect_pose_model.uncollapse()


@reselect_det_settings_button.click
def reselect_det_settings():
    det_settings_editor.readonly = False
    save_det_settings_button.show()
    det_settings_done.hide()
    reselect_det_settings_button.hide()
    card_connect_pose_model.lock()
    card_connect_pose_model.collapse()


@det_redraw_image_button.click
def redraw_det_preview():
    det_labeled_image.hide()
    det_redraw_loading.show()
    preview_det_ann.draw_pretty(
        bitmap=preview_image.copy(),
        output_path=det_preview_output_path,
        thickness=det_line_thickness.get_value(),
        fill_rectangles=False,
    )
    det_labeled_image.set(url="/static/det_labeled.jpg")
    det_labeled_image.show()
    det_redraw_loading.hide()


@connect_pose_model_button.click
def connect_to_pose_model():
    pose_session_id = select_pose_model.get_selected_id()
    if pose_session_id is not None:
        connect_pose_model_button.hide()
        connect_pose_model_done.show()
        select_pose_model.disable()
        pose_model_stats.set_session_id(session_id=pose_session_id)
        pose_model_stats.show()
        change_pose_model_button.show()
        pose_model_meta_json = api.task.send_request(
            pose_session_id,
            "get_output_classes_and_tags",
            data={},
        )
        sly.logger.info(f"Pose estimation model meta: {str(pose_model_meta_json)}")
        pose_model_data["pose_model_meta"] = sly.ProjectMeta.from_json(pose_model_meta_json)
        pose_model_data["pose_session_id"] = pose_session_id
        pose_classes_table.read_meta(pose_model_data["pose_model_meta"])
        # get pose estimation custom inference settings
        pose_inference_settings = api.task.send_request(
            pose_model_data["pose_session_id"],
            "get_custom_inference_settings",
            data={},
        )
        if pose_inference_settings["settings"] is None or len(pose_inference_settings["settings"]) == 0:
            pose_inference_settings["settings"] = ""
        elif isinstance(pose_inference_settings["settings"], dict):
            pose_inference_settings["settings"] = yaml.dump(pose_inference_settings["settings"], allow_unicode=True)
        pose_settings_editor.set_text(pose_inference_settings["settings"])
        card_pose_settings_preview.unlock()
        card_pose_settings_preview.uncollapse()


@change_pose_model_button.click
def change_pose_model():
    select_pose_model.enable()
    connect_pose_model_done.hide()
    pose_model_stats.hide()
    change_pose_model_button.hide()
    connect_pose_model_button.show()
    card_pose_settings_preview.lock()
    card_pose_settings_preview.collapse()


@pose_classes_table.value_changed
def on_pose_classes_selected(selected_pose_classes):
    pose_model_data["pose_model_classes"] = selected_pose_classes
    save_pose_settings_button.show()


@save_pose_settings_button.click
def save_pose_settings():
    pose_classes_table.disable()
    pose_settings_editor.readonly = True
    sly.logger.info(f"Pose estimation model classes: {str(pose_model_data['pose_model_classes'])}")
    save_pose_settings_button.hide()
    pose_settings_done.show()
    pose_preview_loading.show()
    # delete unselected classes
    pose_classes_collection = [cls["title"] for cls in pose_model_data["pose_model_meta"].to_json()["classes"]]
    pose_classes_to_delete = [
        cls for cls in pose_classes_collection if cls not in pose_model_data["pose_model_classes"]
    ]
    pose_model_data["pose_model_meta"] = pose_model_data["pose_model_meta"].delete_obj_classes(pose_classes_to_delete)
    sly.logger.info(f"Updated pose estimation model meta: {str(pose_model_data['pose_model_meta'].to_json())}")
    # get pose estimation inference settings
    pose_inference_settings = pose_settings_editor.get_text()
    pose_inference_settings = yaml.safe_load(pose_inference_settings)
    pose_model_data["pose_inference_settings"] = pose_inference_settings
    if pose_inference_settings is None:
        pose_inference_settings = {}
        sly.logger.info("Pose estimation model doesn't support custom inference settings.")
    else:
        sly.logger.info("Pose estimation inference settings:")
        sly.logger.info(str(pose_inference_settings))
    # merge preview project meta with det model meta
    meta_with_pose = preview_project.meta.merge(pose_model_data["pose_model_meta"])
    preview_project.set_meta(meta_with_pose)
    # get image annotation
    global preview_pose_ann, pose_preview_output_path
    pose_inference_settings["detected_bboxes"] = preview_bboxes
    preview_pose_predictions = api.task.send_request(
        pose_model_data["pose_session_id"],
        "inference_image_id",
        data={"image_id": preview_image_info.id, "settings": pose_inference_settings},
        timeout=500,
    )
    preview_pose_ann = preview_pose_predictions["annotation"]
    preview_pose_ann_objects = preview_pose_ann["objects"].copy()
    for object in preview_pose_ann_objects:
        if object["classTitle"] not in pose_model_data["pose_model_classes"]:
            preview_pose_ann["objects"].remove(object)
    preview_pose_ann = sly.Annotation.from_json(preview_pose_ann, preview_project.meta)
    pose_preview_output_path = os.path.join(static_dir, "pose_labeled.jpg")
    preview_pose_ann.draw_pretty(
        bitmap=preview_image.copy(),
        output_path=pose_preview_output_path,
        thickness=pose_line_thickness.get_value(),
        fill_rectangles=False,
    )
    pose_labeled_image.set(url="/static/pose_labeled.jpg")
    # show preview
    pose_preview_loading.hide()
    card_pose_image_preview.uncollapse()
    card_pose_image_preview.unlock()
    reselect_pose_settings_button.show()
    card_output_project.unlock()
    card_output_project.uncollapse()


@reselect_pose_settings_button.click
def reselect_pose_settings():
    pose_classes_table.enable()
    pose_settings_editor.readonly = False
    pose_classes_table.clear_selection()
    save_pose_settings_button.show()
    pose_settings_done.hide()
    reselect_pose_settings_button.hide()
    card_output_project.lock()
    card_output_project.collapse()


@pose_redraw_image_button.click
def redraw_pose_preview():
    pose_labeled_image.hide()
    pose_redraw_loading.show()
    preview_pose_ann.draw_pretty(
        bitmap=preview_image.copy(),
        output_path=pose_preview_output_path,
        thickness=pose_line_thickness.get_value(),
        fill_rectangles=False,
    )
    pose_labeled_image.set(url="/static/pose_labeled.jpg")
    pose_labeled_image.show()
    pose_redraw_loading.hide()


@apply_models_to_project_button.click
def apply_models_to_project():
    with progress_bar(message="Applying models to project...") as pbar:
        output_project_name_input.enable_readonly()
        # merge output project meta with model metas
        output_project = sly.Project(output_project_dir, mode=sly.OpenMode.READ)
        meta_with_det = output_project.meta.merge(det_model_data["det_model_meta"])
        output_project.set_meta(meta_with_det)
        meta_with_pose = output_project.meta.merge(pose_model_data["pose_model_meta"])
        output_project.set_meta(meta_with_pose)
        output_project_meta = sly.Project(output_project_dir, mode=sly.OpenMode.READ).meta
        # define session ids
        det_session_id = det_model_data["det_session_id"]
        pose_session_id = pose_model_data["pose_session_id"]
        # define inference settings
        det_inference_settings = det_model_data["det_inference_settings"]
        pose_inference_settings = pose_model_data["pose_inference_settings"]
        # define images info
        images_info = []
        datasets_info = {}
        for dataset_info in api.dataset.get_list(project_id):
            images_info.extend(api.image.get_list(dataset_info.id))
            dataset_dir = os.path.join(output_project_dir, dataset_info.name)
            datasets_info[dataset_info.id] = sly.Dataset(dataset_dir, mode=sly.OpenMode.READ)
        # apply models to project
        for image_info in images_info:
            # apply detection model to image
            det_predictions = api.task.send_request(
                det_session_id,
                "inference_image_id",
                data={"image_id": image_info.id, "settings": det_inference_settings},
            )
            # filter detected bboxes according to selected classes
            detected_bboxes = []
            det_annotation = det_predictions["annotation"]
            det_ann_objects = det_annotation["objects"].copy()
            for object in det_ann_objects:
                if object["classTitle"] not in det_model_data["det_model_classes"]:
                    det_annotation["objects"].remove(object)
                else:
                    coordinates = object["points"]["exterior"]
                    det_bbox = {
                        "bbox": [coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], 1.0]
                    }
                    detected_bboxes.append(det_bbox)
            det_annotation = sly.Annotation.from_json(det_annotation, output_project_meta)
            # apply pose estimation model to image
            pose_inference_settings["detected_bboxes"] = detected_bboxes
            pose_predictions = api.task.send_request(
                pose_session_id,
                "inference_image_id",
                data={"image_id": image_info.id, "settings": pose_inference_settings},
            )
            # filter detected keypoints according to selected classes
            pose_annotation = pose_predictions["annotation"]
            for object in pose_annotation["objects"]:
                if object["classTitle"] not in pose_model_data["pose_model_classes"]:
                    pose_annotation["objects"].remove(object)
            pose_annotation = sly.Annotation.from_json(pose_annotation, output_project_meta)
            # merge detection and pose estimation annotations
            total_annotation = det_annotation.add_labels(pose_annotation.labels)
            # annotate image in its dataset
            image_dataset = datasets_info[image_info.dataset_id]
            image_dataset.set_ann(image_info.name, total_annotation)
            pbar.update()
    final_project_id, final_project_name = sly.upload_project(
        dir=output_project_dir,
        api=api,
        workspace_id=workspace_id,
        project_name=output_project_name_input.get_value(),
    )
    final_project_info = api.project.get_info_by_id(final_project_id)
    output_project_thmb.set(info=final_project_info)
    output_project_thmb.show()
    apply_models_to_project_button.hide()
    output_project_done.show()
    sly.logger.info("Project was successfully labeled")

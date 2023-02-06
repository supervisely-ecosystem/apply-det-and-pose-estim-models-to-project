import os
from pathlib import Path
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
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
    Text,
    ModelInfo,
    ClassesTable,
    DoneLabel,
    ProjectThumbnail,
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
det_model_stats_text = Text("Model Info")
det_model_stats_text.hide()
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
        det_model_stats_text,
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


### 4. Detection settings
det_confidence_threshold = InputNumber(value=0.5, min=0, max=1, step=0.1)
det_confidence_threshold_f = Field(det_confidence_threshold, "Detection confidence threshold")
det_iou_threshold = InputNumber(value=0.5, min=0, max=1, step=0.1)
det_iou_threshold_f = Field(det_iou_threshold, "IoU threshold (for NMS)")
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
det_settings_content = Container(
    [
        det_confidence_threshold_f,
        det_iou_threshold_f,
        save_det_settings_button,
        reselect_det_settings_button,
        det_settings_done,
    ]
)
card_det_settings = Card(
    title="Detection Settings",
    content=det_settings_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_det_settings.collapse()
card_det_settings.lock()


### 5. Connect to pose estimation model
select_pose_model = SelectAppSession(allowed_session_tags=["deployed_nn", "deployed_nn_keypoints"])
connect_pose_model_button = Button(
    text='<i style="margin-right: 5px" class="zmdi zmdi-power"></i>connect to pose estimation model',
    button_type="success",
    button_size="small",
)
connect_pose_model_done = DoneLabel("Pose estimation model connected")
connect_pose_model_done.hide()
pose_model_stats_text = Text("Model Info")
pose_model_stats_text.hide()
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
        pose_model_stats_text,
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


### 6. Pose estimation settings
point_confidence_threshold = InputNumber(value=0.01, min=0, max=1, step=0.01)
point_confidence_threshold_f = Field(point_confidence_threshold, "Point confidence threshold")
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
pose_settings_content = Container(
    [
        point_confidence_threshold_f,
        pose_classes_table_f,
        save_pose_settings_button,
        reselect_pose_settings_button,
        pose_settings_done,
    ]
)
card_pose_settings = Card(
    title="Pose Estimation Settings",
    content=pose_settings_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_pose_settings.collapse()
card_pose_settings.lock()


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
            card_det_settings,
            card_connect_pose_model,
            card_pose_settings,
            card_output_project,
        ]
    )
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
        det_model_stats_text.show()
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
    connect_det_model_done.hide()
    det_model_stats_text.hide()
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
    card_det_settings.unlock()
    card_det_settings.uncollapse()


@select_other_det_classes_button.click
def select_other_det_classes():
    det_classes_table.clear_selection()
    select_other_det_classes_button.hide()
    det_classes_done.hide()
    card_det_settings.lock()
    card_det_settings.collapse()


@save_det_settings_button.click
def save_det_settings():
    det_model_data["det_confidence_threshold"] = float(det_confidence_threshold.value)
    det_model_data["det_iou_threshold"] = float(det_iou_threshold.value)
    sly.logger.info(f"Detection confidence threshold: {str(det_model_data['det_confidence_threshold'])}")
    sly.logger.info(f"Detection IoU threshold: {str(det_model_data['det_iou_threshold'])}")
    save_det_settings_button.hide()
    reselect_det_settings_button.show()
    det_settings_done.show()
    card_connect_pose_model.unlock()
    card_connect_pose_model.uncollapse()


@reselect_det_settings_button.click
def reselect_det_settings():
    save_det_settings_button.show()
    det_settings_done.hide()
    reselect_det_settings_button.hide()
    card_connect_pose_model.lock()
    card_connect_pose_model.collapse()


@connect_pose_model_button.click
def connect_to_pose_model():
    pose_session_id = select_pose_model.get_selected_id()
    if pose_session_id is not None:
        connect_pose_model_button.hide()
        connect_pose_model_done.show()
        pose_model_stats.set_session_id(session_id=pose_session_id)
        pose_model_stats_text.show()
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
        card_pose_settings.unlock()
        card_pose_settings.uncollapse()


@change_pose_model_button.click
def change_pose_model():
    connect_pose_model_done.hide()
    pose_model_stats_text.hide()
    pose_model_stats.hide()
    change_pose_model_button.hide()
    connect_pose_model_button.show()
    card_pose_settings.lock()
    card_pose_settings.collapse()


@pose_classes_table.value_changed
def on_pose_classes_selected(selected_pose_classes):
    pose_model_data["pose_model_classes"] = selected_pose_classes
    save_pose_settings_button.show()


@save_pose_settings_button.click
def save_pose_settings():
    pose_model_data["pose_confidence_threshold"] = float(point_confidence_threshold.value)
    sly.logger.info(f"Pose estimation model classes: {str(pose_model_data['pose_model_classes'])}")
    sly.logger.info(f"Point confidence threshold: {str(pose_model_data['pose_confidence_threshold'])}")
    save_pose_settings_button.hide()
    reselect_pose_settings_button.show()
    pose_settings_done.show()
    # delete unselected classes
    pose_classes_collection = [cls["title"] for cls in pose_model_data["pose_model_meta"].to_json()["classes"]]
    pose_classes_to_delete = [
        cls for cls in pose_classes_collection if cls not in pose_model_data["pose_model_classes"]
    ]
    pose_model_data["pose_model_meta"] = pose_model_data["pose_model_meta"].delete_obj_classes(pose_classes_to_delete)
    sly.logger.info(f"Updated pose estimation model meta: {str(pose_model_data['pose_model_meta'].to_json())}")
    card_output_project.unlock()
    card_output_project.uncollapse()


@reselect_pose_settings_button.click
def reselect_pose_settings():
    pose_classes_table.clear_selection()
    save_pose_settings_button.show()
    pose_settings_done.hide()
    reselect_pose_settings_button.hide()
    card_output_project.lock()
    card_output_project.collapse()


@apply_models_to_project_button.click
def apply_models_to_project():
    with progress_bar(message="Applying models to project...") as pbar:
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
        det_inference_settings = {
            "conf_thres": det_model_data["det_confidence_threshold"],
            "iou_thres": det_model_data["det_iou_threshold"],
            "inference_mode": "full_image",
        }
        pose_inference_settings = {"point_threshold": pose_model_data["pose_confidence_threshold"]}
        # define images info
        images_info = []
        datasets_info = {}
        for dataset_info in api.dataset.get_list(project_id):
            images_info.extend(api.image.get_list(dataset_info.id))
            dataset_dir = os.path.join(output_project_dir, dataset_info.name)
            datasets_info[dataset_info.id] = sly.Dataset(dataset_dir, mode=sly.OpenMode.READ)
        # apply models to project
        for image_info in images_info:
            print(image_info)
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

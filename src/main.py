import os
import random
import supervisely as sly
import src.globals as g
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
    Editor,
    Select,
    Checkbox,
    RadioGroup,
)


# function for updating global variables
def update_globals(new_dataset_ids):
    global dataset_ids, project_id, workspace_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids == [None]:
        dataset_ids = None
    if dataset_ids:
        dataset = api.dataset.get_info_by_id(dataset_ids[0])
        if dataset is None:
            sly.app.show_dialog(
                title="Dataset not found",
                description="Please, please select another dataset or reload the page and try again",
                status="error",
            )
            return
        project_id = dataset.project_id
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            sly.app.show_dialog(
                title="Project not found",
                description="Please, please select another project or reload the page and try again",
                status="error",
            )
            return
        workspace_id = project_info.workspace_id
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        print(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            sly.app.show_dialog(
                title="Project not found",
                description="Please, please select another project or reload the page and try again",
                status="error",
            )
            return
        workspace_id = project_info.workspace_id
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        dataset_ids = [dataset_info.id for dataset_info in api.dataset.get_list(project_id)]
    else:
        print("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, project_info, project_meta = [None] * 4


# authentication
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
team_id = sly.env.team_id()

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)

sly.logger.info(f"App root directory: {g.app_root_directory}")
# create directory for storing preview files
os.makedirs(g.static_dir, exist_ok=True)
sly.io.fs.clean_dir(g.static_dir)
# dictionaries for storing detection and pose estimation model data
det_model_data = {}
pose_model_data = {}


### 1. Dataset selection
dataset_selector = SelectDataset(project_id=project_id, multiselect=True, select_all_datasets=True, allowed_project_types=[sly.ProjectType.IMAGES])
select_data_button = Button("Select data")
select_done = DoneLabel("Successfully selected input data")
select_done.hide()
reselect_data_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Reselect data',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_data_button.hide()
project_settings_content = Container(
    [
        dataset_selector,
        select_data_button,
        select_done,
        reselect_data_button,
    ]
)
card_project_settings = Card(title="Dataset selection", content=project_settings_content)


### 2. Method of getting bounding boxes selection
select_det_method = RadioGroup(
    items=[
        RadioGroup.Item(value="use pretrained detection model to label images with bounding boxes"),
        RadioGroup.Item(value="use existing bounding boxes if images are already labeled with bounding boxes"),
    ],
    direction="vertical",
)
det_existing_classes_table = ClassesTable()
det_existing_classes_table_f = Field(det_existing_classes_table, "Select which classes from project to use")
det_existing_classes_table_f.hide()
select_det_method_button = Button("Save")
select_det_method_done = DoneLabel("Successfully selected method of getting bounding boxes")
select_det_method_done.hide()
change_det_method_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>change method of getting bounding boxes',
    button_type="warning",
    button_size="small",
    plain=True,
)
change_det_method_button.hide()
select_det_method_content = Container(
    [
        select_det_method,
        det_existing_classes_table_f,
        select_det_method_button,
        select_det_method_done,
        change_det_method_button,
    ]
)
card_select_det_method = Card(
    title="Select method of getting bounding boxes",
    description="You can use pretrained object detection model or existing annotation",
    content=select_det_method_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_select_det_method.collapse()
card_select_det_method.lock()


### 3. Connect to detection model
select_det_model = SelectAppSession(team_id=team_id, tags=["deployed_nn"])
connect_det_model_button = Button(
    text='<i style="margin-right: 5px" class="zmdi zmdi-power"></i>connect to detection model',
    button_type="success",
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


### 4. Detection model classes
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


### 5.1 Detection settings
det_settings_editor = Editor(language_mode="yaml", height_lines=30)
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
        det_settings_editor,
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


### 5.2 Detection inference preview
det_line_thickness = InputNumber(value=7, min=1, max=14)
det_line_thickness_f = Field(det_line_thickness, "Line thickness")
det_redraw_image_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>redraw',
    button_type="warning",
    button_size="small",
    plain=True,
)
det_line_settings = Container([det_line_thickness_f, det_redraw_image_button])
select_det_preview = Select(items=[Select.Item(value="Random image")])
select_det_preview.disable()
select_det_preview_f = Field(select_det_preview, "Select image for preview")
det_is_random_preview = Checkbox(content="Random image", checked=True)
det_preview_params = Container(
    [det_line_settings, select_det_preview_f, det_is_random_preview],
    direction="horizontal",
    fractions=[1, 1, 1],
)
det_labeled_image = Image()
det_image_preview_content = Container(
    [
        det_preview_params,
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
    fractions=[1, 1],
)


### 6. Connect to pose estimation model
select_pose_model = SelectAppSession(team_id=team_id, tags=["deployed_nn_keypoints"])
connect_pose_model_button = Button(
    text='<i style="margin-right: 5px" class="zmdi zmdi-power"></i>connect to pose estimation model',
    button_type="success",
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


### 7. Pose estimation model classes
pose_classes_table = ClassesTable()
select_pose_classes_button = Button("select classes")
select_pose_classes_button.hide()
select_other_pose_classes_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>select other classes',
    button_type="warning",
    button_size="small",
    plain=True,
)
select_other_pose_classes_button.hide()
pose_classes_done = DoneLabel()
pose_classes_done.hide()
pose_model_classes_content = Container(
    [
        pose_classes_table,
        select_pose_classes_button,
        select_other_pose_classes_button,
        pose_classes_done,
    ]
)
card_pose_model_classes = Card(
    title="Pose Estimation Model Classes",
    description="Choose classes that will be kept after prediction, other classes will be ignored",
    content=pose_model_classes_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_pose_model_classes.collapse()
card_pose_model_classes.lock()


### 8.1 Pose estimation settings
pose_settings_editor = Editor(language_mode="yaml", height_lines=30)
save_pose_settings_button = Button("save pose estimation settings")
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
        pose_settings_editor,
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


### 8.2 Pose estimation inference preview
pose_line_thickness = InputNumber(value=7, min=1, max=14)
pose_line_thickness_f = Field(pose_line_thickness, "Line thickness")
pose_redraw_image_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>redraw',
    button_type="warning",
    button_size="small",
    plain=True,
)
pose_line_settings = Container([pose_line_thickness_f, pose_redraw_image_button])
select_pose_preview = Select(items=[Select.Item(value="Random image")])
select_pose_preview.disable()
select_pose_preview_f = Field(select_pose_preview, "Select image for preview")
pose_is_random_preview = Checkbox(content="Random image", checked=True)
pose_preview_params = Container(
    [pose_line_settings, select_pose_preview_f, pose_is_random_preview],
    direction="horizontal",
    fractions=[1, 1, 1],
)
pose_labeled_image = Image()
pose_image_preview_content = Container(
    [
        pose_preview_params,
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
    fractions=[1, 1],
)


### 9. Output project
output_project_name_input = Input(value="Labeled project")
output_project_name_input_f = Field(output_project_name_input, "Output project name")
apply_models_to_project_button = Button("APPLY MODELS TO PROJECT")
apply_progress_bar = Progress()
output_project_thmb = ProjectThumbnail()
output_project_thmb.hide()
output_project_done = DoneLabel("done")
output_project_done.hide()
output_project_content = Container(
    [
        output_project_name_input_f,
        apply_models_to_project_button,
        apply_progress_bar,
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
            card_select_det_method,
            card_connect_det_model,
            card_det_model_classes,
            det_settings_preview_content,
            card_connect_pose_model,
            card_pose_model_classes,
            pose_settings_preview_content,
            card_output_project,
        ]
    ),
    static_dir=g.static_dir,
)


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    if new_dataset_ids == []:
        select_data_button.hide()
    elif new_dataset_ids != [] and select_data_button.is_hidden():
        select_data_button.show()
    update_globals(new_dataset_ids)
    if project_info is not None:
        # set default output project name
        output_project_name = project_info.name + " (keypoints prediction)"
        output_project_name_input.set_value(value=api.project.get_free_name(workspace_id, output_project_name))


@select_data_button.click
def download_input_data():
    global dataset_ids, project_id, workspace_id
    select_data_button.loading = True
    dataset_selector.disable()
    # download input project to ouput project directory
    if os.path.exists(g.output_project_dir):
        sly.fs.clean_dir(g.output_project_dir)
    if dataset_ids is None or dataset_ids == []:
        proj_id = dataset_selector.get_selected_project_id()
        if project_id is None:
            project_id = proj_id
        if proj_id:
            dataset_ids = [dataset_info.id for dataset_info in api.dataset.get_list(project_id)]
    project_info = api.project.get_info_by_id(project_id)
    if project_info is None:
        sly.app.show_dialog(
            title="Project not found",
            description="Please, please select another project or reload the page and try again",
            status="error",
        )
        select_data_button.loading = False
        dataset_selector.enable()
        return
    if workspace_id is None:
        workspace_id = project_info.workspace_id
    sly.download_project(
        api=api,
        project_id=project_id,
        dest_dir=g.output_project_dir,
        dataset_ids=dataset_ids,
        log_progress=True,
        save_image_info=True,
        save_images=False,
    )
    select_data_button.loading = False
    select_data_button.hide()
    select_done.show()
    reselect_data_button.show()
    # card_connect_det_model.unlock()
    # card_connect_det_model.uncollapse()
    card_select_det_method.unlock()
    card_select_det_method.uncollapse()


@reselect_data_button.click
def redownload_input_data():
    select_data_button.show()
    reselect_data_button.hide()
    select_done.hide()
    dataset_selector.enable()


@select_det_method.value_changed
def change_det_method(value):
    if value == "use existing bounding boxes if images are already labeled with bounding boxes":
        det_existing_classes_table.read_meta(project_meta)
        det_existing_classes_table_f.show()
        card_connect_det_model.hide()
        card_det_model_classes.hide()
        det_settings_preview_content.hide()
    else:
        card_connect_det_model.show()
        card_det_model_classes.show()
        det_settings_preview_content.show()
        det_existing_classes_table_f.hide()


@select_det_method_button.click
def det_method_select():
    method = select_det_method.get_value()
    problem = False
    if method == "use pretrained detection model to label images with bounding boxes":
        card_connect_det_model.unlock()
        card_connect_det_model.uncollapse()
    else:
        selected_classes = det_existing_classes_table.get_selected_classes()
        if len(selected_classes) < 1:
            sly.app.show_dialog(
                title="At least 1 class must be selected",
                description="Please, select at least 1 class of shape rectangle in the classes table",
                status="warning",
            )
            problem = True
        else:
            selected_shapes = [cls.geometry_type.geometry_name() for cls in project_meta.obj_classes if cls.name in selected_classes]
            if "rectangle" not in selected_shapes:
                sly.app.show_dialog(
                    title="There are no classes of shape rectangle in the list of selected classes",
                    description="Please, select at least 1 class of shape rectangle or change input data",
                    status="warning",
                )
                problem = True
            else:
                card_connect_pose_model.unlock()
                card_connect_pose_model.uncollapse()
                det_existing_classes_table.disable()
    if not problem:
        select_det_method_button.hide()
        select_det_method_done.show()
        change_det_method_button.show()


@change_det_method_button.click
def det_method_reselect():
    method = select_det_method.get_value()
    if method == "use pretrained detection model to label images with bounding boxes":
        card_connect_det_model.lock()
        card_connect_det_model.collapse()
    else:
        card_connect_pose_model.lock()
        card_connect_pose_model.collapse()
    select_det_method_button.show()
    select_det_method_done.hide()
    change_det_method_button.hide()
    det_existing_classes_table.enable()


@connect_det_model_button.click
def connect_to_det_model():
    det_session_id = select_det_model.get_selected_id()
    if det_session_id is not None:
        connect_det_model_button.hide()
        connect_det_model_done.show()
        select_det_model.disable()
        # show detection model info
        det_model_stats.set_session_id(session_id=det_session_id)
        det_model_stats.show()
        change_det_model_button.show()
        # get detection model meta
        det_model_meta_json = api.task.send_request(
            det_session_id,
            "get_output_classes_and_tags",
            data={},
        )
        sly.logger.info(f"Detection model meta: {str(det_model_meta_json)}")
        det_model_data["det_model_meta"] = sly.ProjectMeta.from_json(det_model_meta_json)
        det_model_data["det_session_id"] = det_session_id
        # show detection classes table
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
    if n_det_classes > 0:
        if n_det_classes > 1:
            select_det_classes_button.text = f"Select {n_det_classes} classes"
        else:
            select_det_classes_button.text = f"Select {n_det_classes} class"
        select_det_classes_button.show()
    else:
        select_det_classes_button.hide()


# function for getting random image from selected project
def get_random_image(images_info):
    image_idx = random.randint(0, len(images_info) - 1)
    random_image_info = images_info[image_idx]
    return random_image_info


# function for drawing inference previews
def draw_inference_preview(image_info, mode, det_settings, pose_settings=None):
    method = select_det_method.get_value()
    # get det annotation
    if method == "use pretrained detection model to label images with bounding boxes":
        preview_det_predictions = api.task.send_request(
            det_model_data["det_session_id"],
            "inference_image_id",
            data={"image_id": image_info.id, "settings": det_settings},
            timeout=500,
        )
        preview_det_ann = preview_det_predictions["annotation"]
        det_classes = det_model_data["det_model_classes"]
    else:
        preview_det_ann = api.annotation.download_json(image_info.id)
        det_classes = det_existing_classes_table.get_selected_classes()
    preview_det_ann_objects = preview_det_ann["objects"].copy()
    # filter object classes in annotation according to selected classes
    preview_bboxes = []
    
    for object in preview_det_ann_objects:
        if object["classTitle"] not in det_classes:
            preview_det_ann["objects"].remove(object)
        else:
            # save predicted bounding boxes to detect keypoints inside them using pose estimation model
            coordinates = object["points"]["exterior"]
            det_bbox = {"bbox": [coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], 1.0]}
            preview_bboxes.append(det_bbox)
    preview_det_ann = sly.Annotation.from_json(preview_det_ann, preview_project_meta)
    if mode == "det":
        preview_image = api.image.download_np(image_info.id)
        # draw predicted bounding boxes on preview image
        preview_det_ann.draw_pretty(
            bitmap=preview_image.copy(),
            output_path=g.local_det_preview_path,
            thickness=det_line_thickness.get_value(),
            fill_rectangles=False,
        )
        # upload detection inference preview image to team files
        det_preview_file_info = api.file.upload(team_id, g.local_det_preview_path, g.remote_det_preview_path)
        det_labeled_image.set(url=det_preview_file_info.full_storage_url)
    elif mode == "pose":
        pose_settings["detected_bboxes"] = preview_bboxes
        # detect keypoints on image using bounding boxes predicted by detetction model
        preview_pose_predictions = api.task.send_request(
            pose_model_data["pose_session_id"],
            "inference_image_id",
            data={"image_id": image_info.id, "settings": pose_settings},
            timeout=500,
        )
        preview_pose_ann = preview_pose_predictions["annotation"]
        preview_pose_ann_objects = preview_pose_ann["objects"].copy()
        # filter object classes in annotation according to selected classes
        for object in preview_pose_ann_objects:
            if (
                object["classTitle"] not in pose_model_data["pose_model_classes"]
                and object["classTitle"] != "animal_keypoints"
            ):
                preview_pose_ann["objects"].remove(object)
        preview_pose_ann = sly.Annotation.from_json(preview_pose_ann, preview_project_meta)
        # merge detection and pose estimation annotations
        total_ann = preview_det_ann.add_labels(preview_pose_ann.labels)
        preview_image = api.image.download_np(image_info.id)
        # draw predicted keypoints graph on image
        total_ann.draw_pretty(
            bitmap=preview_image.copy(),
            output_path=g.local_pose_preview_path,
            thickness=pose_line_thickness.get_value(),
            fill_rectangles=False,
        )
        # upload pose estimation inference preview image to team files
        pose_preview_file_info = api.file.upload(
            team_id,
            g.local_pose_preview_path,
            g.remote_pose_preview_path,
        )
        pose_labeled_image.set(url=pose_preview_file_info.full_storage_url)


@select_det_classes_button.click
def select_det_classes():
    det_classes_table.disable()
    # get selected classes for detection model
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
    # delete unselected classes from detection model meta
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
    det_model_data["det_inference_settings"] = det_inference_settings
    if det_inference_settings["settings"] is None or len(det_inference_settings["settings"]) == 0:
        det_inference_settings["settings"] = ""
    elif isinstance(det_inference_settings["settings"], dict):
        det_inference_settings["settings"] = yaml.dump(det_inference_settings["settings"], allow_unicode=True)
    det_settings_editor.set_text(det_inference_settings["settings"])
    card_det_settings.unlock()
    card_det_settings.uncollapse()
    # create preview project meta
    card_det_image_preview.loading = True
    global preview_project_meta, images_info
    preview_project_meta = api.project.get_meta(id=project_id)
    preview_project_meta = sly.ProjectMeta.from_json(preview_project_meta)
    # merge preview project meta with det model meta
    preview_project_meta = preview_project_meta.merge(det_model_data["det_model_meta"])
    # define images info
    images_info = []
    for dataset_info in api.dataset.get_list(project_id):
        if dataset_ids and dataset_info != [None]:
            if dataset_info.id not in dataset_ids:
                continue
        images_info.extend(api.image.get_list(dataset_info.id))
    preview_image_info = get_random_image(images_info)
    # draw detection preview
    draw_inference_preview(preview_image_info, mode="det", det_settings=det_inference_settings)
    card_det_image_preview.loading = False
    card_det_image_preview.uncollapse()
    card_det_image_preview.unlock()


@select_other_det_classes_button.click
def select_other_det_classes():
    det_classes_table.enable()
    det_classes_table.clear_selection()
    select_other_det_classes_button.hide()
    det_classes_done.hide()
    card_det_settings.lock()
    card_det_settings.collapse()
    card_det_image_preview.lock()
    card_det_image_preview.collapse()


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

    reselect_det_settings_button.show()
    card_connect_pose_model.unlock()
    card_connect_pose_model.uncollapse()


@det_is_random_preview.value_changed
def select_det_preview_image(value):
    if value is False:
        select_det_preview.enable()
        select_det_preview.set(items=[Select.Item(image_info.id, image_info.name) for image_info in images_info])
    else:
        select_det_preview.disable()
        select_det_preview.set(items=[Select.Item(value="Random image")])


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
    card_det_image_preview.loading = True
    if det_is_random_preview.is_checked():
        preview_image_info = get_random_image(images_info)
    else:
        id = select_det_preview.get_value()
        preview_image_info = api.image.get_info_by_id(id=id)
    det_inference_settings = det_settings_editor.get_text()
    det_inference_settings = yaml.safe_load(det_inference_settings)
    draw_inference_preview(preview_image_info, mode="det", det_settings=det_inference_settings)
    det_labeled_image.show()
    card_det_image_preview.loading = False


@connect_pose_model_button.click
def connect_to_pose_model():
    pose_session_id = select_pose_model.get_selected_id()
    if pose_session_id is not None:
        connect_pose_model_button.hide()
        connect_pose_model_done.show()
        select_pose_model.disable()
        # show pose estimation model info
        pose_model_stats.set_session_id(session_id=pose_session_id)
        pose_model_stats.show()
        change_pose_model_button.show()
        # get pose estimation model meta
        pose_model_meta_json = api.task.send_request(
            pose_session_id,
            "get_output_classes_and_tags",
            data={},
        )
        sly.logger.info(f"Pose estimation model meta: {str(pose_model_meta_json)}")
        pose_model_data["pose_model_meta"] = sly.ProjectMeta.from_json(pose_model_meta_json)
        pose_model_data["pose_session_id"] = pose_session_id
        # show pose estimation classes table
        pose_classes_table.read_meta(pose_model_data["pose_model_meta"])
        card_pose_model_classes.uncollapse()
        card_pose_model_classes.unlock()


@change_pose_model_button.click
def change_pose_model():
    select_pose_model.enable()
    connect_pose_model_done.hide()
    pose_model_stats.hide()
    change_pose_model_button.hide()
    connect_pose_model_button.show()
    card_pose_model_classes.lock()
    card_pose_model_classes.collapse()


@pose_classes_table.value_changed
def on_pose_classes_selected(selected_pose_classes):
    n_pose_classes = len(selected_pose_classes)
    if n_pose_classes > 0:
        if n_pose_classes > 1:
            select_pose_classes_button.text = f"Select {n_pose_classes} classes"
        else:
            select_pose_classes_button.text = f"Select {n_pose_classes} class"
        select_pose_classes_button.show()
    else:
        select_pose_classes_button.hide()


@select_pose_classes_button.click
def select_pose_classes():
    pose_model_data["pose_model_classes"] = pose_classes_table.get_selected_classes()
    n_pose_classes = len(pose_model_data["pose_model_classes"])
    pose_classes_table.disable()
    sly.logger.info(f"Pose estimation model classes: {str(pose_model_data['pose_model_classes'])}")
    select_pose_classes_button.hide()
    if n_pose_classes > 1:
        pose_classes_done.text = f"{n_pose_classes} classes were selected successfully"
    else:
        pose_classes_done.text = f"{n_pose_classes} class was selected successfully"
    pose_classes_done.show()
    select_other_pose_classes_button.show()
    # delete unselected classes
    pose_classes_collection = [
        cls["title"]
        for cls in pose_model_data["pose_model_meta"].to_json()["classes"]
        if cls["title"] != "animal_keypoints"
    ]
    pose_classes_to_delete = [
        cls for cls in pose_classes_collection if cls not in pose_model_data["pose_model_classes"]
    ]
    pose_model_data["pose_model_meta"] = pose_model_data["pose_model_meta"].delete_obj_classes(pose_classes_to_delete)
    sly.logger.info(f"Updated pose estimation model meta: {str(pose_model_data['pose_model_meta'].to_json())}")
    # get pose estimation custom inference settings
    pose_inference_settings = api.task.send_request(
        pose_model_data["pose_session_id"],
        "get_custom_inference_settings",
        data={},
    )
    pose_model_data["pose_inference_settings"] = pose_inference_settings
    if pose_inference_settings["settings"] is None or len(pose_inference_settings["settings"]) == 0:
        pose_inference_settings["settings"] = ""
    elif isinstance(pose_inference_settings["settings"], dict):
        pose_inference_settings["settings"] = yaml.dump(pose_inference_settings["settings"], allow_unicode=True)
    pose_settings_editor.set_text(pose_inference_settings["settings"])
    card_pose_settings.unlock()
    card_pose_settings.uncollapse()
    card_pose_image_preview.loading = True
    # merge preview project meta with pose model meta
    global preview_project_meta, images_info
    if select_det_method.get_value() == "use existing bounding boxes if images are already labeled with bounding boxes":
        preview_project_meta = project_meta
        images_info = []
        for dataset_info in api.dataset.get_list(project_id):
            if dataset_ids and dataset_info != [None]:
                if dataset_info.id not in dataset_ids:
                    continue
            images_info.extend(api.image.get_list(dataset_info.id))
        det_model_data["det_inference_settings"] = {}
    preview_project_meta = preview_project_meta.merge(pose_model_data["pose_model_meta"])
    preview_image_info = get_random_image(images_info)
    # draw pose estimation preview
    draw_inference_preview(
        preview_image_info,
        mode="pose",
        det_settings=det_model_data["det_inference_settings"],
        pose_settings=pose_inference_settings,
    )
    card_pose_image_preview.loading = False
    card_pose_image_preview.uncollapse()
    card_pose_image_preview.unlock()


@select_other_pose_classes_button.click
def select_other_pose_classes():
    pose_classes_table.enable()
    pose_classes_table.clear_selection()
    select_other_pose_classes_button.hide()
    pose_classes_done.hide()
    card_pose_settings.lock()
    card_pose_settings.collapse()
    card_pose_image_preview.lock()
    card_pose_image_preview.collapse()


@save_pose_settings_button.click
def save_pose_settings():
    pose_settings_editor.readonly = True
    save_pose_settings_button.hide()
    pose_settings_done.show()
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
    # detect keypoints on image using bounding boxes predicted by detetction model

    reselect_pose_settings_button.show()
    card_output_project.unlock()
    card_output_project.uncollapse()


@pose_is_random_preview.value_changed
def select_pose_preview_image(value):
    if value is False:
        select_pose_preview.enable()
        select_pose_preview.set(items=[Select.Item(image_info.id, image_info.name) for image_info in images_info])
    else:
        select_pose_preview.disable()
        select_pose_preview.set(items=[Select.Item(value="Random image")])


@reselect_pose_settings_button.click
def reselect_pose_settings():
    save_pose_settings_button.show()
    pose_settings_editor.readonly = False
    pose_settings_done.hide()
    reselect_pose_settings_button.hide()
    card_output_project.lock()
    card_output_project.collapse()


@pose_redraw_image_button.click
def redraw_pose_preview():
    pose_labeled_image.hide()
    card_pose_image_preview.loading = True
    if pose_is_random_preview.is_checked():
        preview_image_info = get_random_image(images_info)
    else:
        id = select_pose_preview.get_value()
        preview_image_info = api.image.get_info_by_id(id=id)
    pose_inference_settings = pose_settings_editor.get_text()
    pose_inference_settings = yaml.safe_load(pose_inference_settings)
    draw_inference_preview(
        preview_image_info,
        mode="pose",
        det_settings=det_model_data["det_inference_settings"],
        pose_settings=pose_inference_settings,
    )
    pose_labeled_image.show()
    card_pose_image_preview.loading = False


@apply_models_to_project_button.click
def apply_models_to_project():
    method = select_det_method.get_value()
    apply_models_to_project_button.loading = True
    output_project_name_input.enable_readonly()
    output_project = sly.Project(g.output_project_dir, mode=sly.OpenMode.READ)
    # merge output project meta with model metas
    global images_info
    if method == "use pretrained detection model to label images with bounding boxes":
        meta_with_det = output_project.meta.merge(det_model_data["det_model_meta"])
    else:
        meta_with_det = project_meta
        images_info = []
        for dataset_info in api.dataset.get_list(project_id):
            if dataset_ids and dataset_info != [None]:
                if dataset_info.id not in dataset_ids:
                    continue
            images_info.extend(api.image.get_list(dataset_info.id))
    output_project.set_meta(meta_with_det)
    meta_with_pose = output_project.meta.merge(pose_model_data["pose_model_meta"])
    output_project.set_meta(meta_with_pose)
    output_project_meta = sly.Project(g.output_project_dir, mode=sly.OpenMode.READ).meta
    if method == "use pretrained detection model to label images with bounding boxes":
        det_session_id = det_model_data["det_session_id"]
    pose_session_id = pose_model_data["pose_session_id"]
    # define inference settings
    det_inference_settings = det_model_data["det_inference_settings"]
    pose_inference_settings = pose_model_data["pose_inference_settings"]
    # get datasets info
    datasets_info = {}
    for dataset_info in api.dataset.get_list(project_id):
        if dataset_ids and dataset_info != [None]:
            if dataset_info.id not in dataset_ids:
                continue
        dataset_dir = os.path.join(g.output_project_dir, dataset_info.name)
        datasets_info[dataset_info.id] = sly.Dataset(dataset_dir, mode=sly.OpenMode.READ)
    # apply models to project
    with apply_progress_bar(message="Applying models to project...", total=len(images_info)) as pbar:
        for image_info in images_info:
            if method == "use pretrained detection model to label images with bounding boxes":
                # apply detection model to image
                det_predictions = api.task.send_request(
                    det_session_id,
                    "inference_image_id",
                    data={"image_id": image_info.id, "settings": det_inference_settings},
                )
                det_annotation = det_predictions["annotation"]
                det_classes = det_model_data["det_model_classes"]
            else:
                det_annotation = api.annotation.download_json(image_info.id)
                det_classes = det_existing_classes_table.get_selected_classes()
            # filter detected bboxes according to selected classes
            detected_bboxes = []
            detected_classes = []
            det_ann_objects = det_annotation["objects"].copy()
            for object in det_ann_objects:
                if object["classTitle"] not in det_classes:
                    det_annotation["objects"].remove(object)
                else:
                    detected_classes.append(object["classTitle"])
                    coordinates = object["points"]["exterior"]
                    det_bbox = {
                        "bbox": [coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], 1.0]
                    }
                    detected_bboxes.append(det_bbox)
            det_annotation = sly.Annotation.from_json(det_annotation, output_project_meta)
            # apply pose estimation model to image
            pose_inference_settings["detected_bboxes"] = detected_bboxes
            pose_inference_settings["detected_classes"] = detected_classes
            pose_predictions = api.task.send_request(
                pose_session_id,
                "inference_image_id",
                data={"image_id": image_info.id, "settings": pose_inference_settings},
            )
            # filter detected keypoints according to selected classes
            pose_annotation = pose_predictions["annotation"]
            for object in pose_annotation["objects"]:
                if (
                    object["classTitle"] not in pose_model_data["pose_model_classes"]
                    and object["classTitle"] != "animal_keypoints"
                ):
                    pose_annotation["objects"].remove(object)
            pose_annotation = sly.Annotation.from_json(pose_annotation, output_project_meta)
            # merge detection and pose estimation annotations
            total_annotation = det_annotation.add_labels(pose_annotation.labels)
            # annotate image in its dataset
            image_dataset = datasets_info[image_info.dataset_id]
            image_dataset.set_ann(image_info.name, total_annotation)
            pbar.update()
    # upload labeled project to platform
    final_project_id, final_project_name = sly.upload_project(
        dir=g.output_project_dir,
        api=api,
        workspace_id=workspace_id,
        project_name=output_project_name_input.get_value(),
        log_progress=True,
    )
    # prepare project thumbnail
    final_project_info = api.project.get_info_by_id(final_project_id)
    output_project_thmb.set(info=final_project_info)
    output_project_thmb.show()
    apply_models_to_project_button.loading = False
    apply_models_to_project_button.hide()
    output_project_done.show()
    # remove unnecessary files and directories since they are no longer needed
    api.file.remove(team_id, "/" + g.remote_det_preview_path)
    api.file.remove(team_id, "/" + g.remote_pose_preview_path)
    sly.io.fs.remove_dir(g.app_data_dir)
    sly.logger.info("Project was successfully labeled")
    app.stop()

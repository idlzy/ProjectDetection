from pathlib import Path

from tools.infer import bev_path_for, collect_images, output_paths


def test_collect_folder_images_and_build_paired_outputs(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    first = input_dir / "a.png"
    second = input_dir / "b.jpg"
    ignored = input_dir / "notes.txt"
    first.touch()
    second.touch()
    ignored.touch()

    assert collect_images(input_dir, recursive=False) == [first, second]
    camera, bev = output_paths(
        first, input_dir, tmp_path / "results", multiple=True
    )
    assert camera == tmp_path / "results" / "a_pred.jpg"
    assert bev == tmp_path / "results" / "a_bev.jpg"


def test_single_image_bev_path_preserves_existing_output():
    camera = Path("outputs/sample_pred.jpg")
    assert bev_path_for(camera) == Path("outputs/sample_bev.jpg")

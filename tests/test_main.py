import unittest
from unittest.mock import patch

import main


class MainTest(unittest.TestCase):
    def test_main_runs_pipeline_by_default(self):
        with patch("main.run") as run, patch("main.select_line_from_source") as select_line:
            main.main(["--config", "config/custom.yaml"])

        run.assert_called_once_with("config/custom.yaml")
        select_line.assert_not_called()

    def test_main_passes_debug_options_to_pipeline(self):
        with patch("main.run") as run:
            main.main(["--config", "config/custom.yaml", "--debug", "--debug-every", "5"])

        run.assert_called_once_with("config/custom.yaml", debug=True, debug_every=5)

    def test_main_passes_model_path_override_to_pipeline(self):
        with patch("main.run") as run:
            main.main(["--config", "config/custom.yaml", "--model-path", "models/inference_model.onnx"])

        run.assert_called_once_with("config/custom.yaml", model_path="models/inference_model.onnx")

    def test_main_combines_model_path_override_with_debug_options(self):
        with patch("main.run") as run:
            main.main(
                [
                    "--config",
                    "config/custom.yaml",
                    "--model-path",
                    "models/inference_model.onnx",
                    "--debug",
                    "--debug-every",
                    "5",
                    "--debug-video",
                ]
            )

        run.assert_called_once_with(
            "config/custom.yaml",
            model_path="models/inference_model.onnx",
            debug=True,
            debug_every=5,
            debug_video=True,
        )

    def test_main_passes_debug_video_option_to_pipeline(self):
        with patch("main.run") as run:
            main.main(["--config", "config/custom.yaml", "--debug-video"])

        run.assert_called_once_with("config/custom.yaml", debug_video=True)

    def test_main_combines_text_debug_and_debug_video_options(self):
        with patch("main.run") as run:
            main.main(["--config", "config/custom.yaml", "--debug", "--debug-every", "5", "--debug-video"])

        run.assert_called_once_with("config/custom.yaml", debug=True, debug_every=5, debug_video=True)

    def test_main_starts_line_selection_when_flag_is_present(self):
        with patch("main.run") as run, patch("main.load_config") as load_config, patch(
            "main.select_line_from_source"
        ) as select_line:
            load_config.return_value = {"cameras": [{"source": "videos/gate_01.mp4", "line_width": 60.0}]}

            main.main(["--select-line", "videos/gate_01.mp4", "--model-path", "models/inference_model.onnx"])

        load_config.assert_called_once_with("config/cameras.yaml")
        select_line.assert_called_once_with("videos/gate_01.mp4", line_width=60.0)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

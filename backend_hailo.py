"""Hailo inference backend."""

import contextlib

import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
)

import monitor
from detector import preprocess, postprocess


class HailoBackend:
    """Hailo-8 inference backend (context manager).

    Parameters
    ----------
    model_path : str
        Path to the compiled .hef model file.
    batch_size : int
        Must match the batch size the HEF was compiled with.
    conf_thresh : float
        Minimum confidence score for detections.
    """

    def __init__(self, model_path: str, batch_size: int, conf_thresh: float):
        self._model_path = model_path
        self._batch_size = batch_size
        self._conf_thresh = conf_thresh
        self._stack = contextlib.ExitStack()
        self._pipeline = None
        self._input_name = None
        self._output_name = None
        self._hailo_mon = None
        self.input_height: int = 0
        self.input_width: int = 0
        self.max_tiles: int | None = None

    def __enter__(self):
        print(f"Loading HEF: {self._model_path}")
        hef = HEF(str(self._model_path))
        input_infos = hef.get_input_vstream_infos()
        output_infos = hef.get_output_vstream_infos()

        input_shape = input_infos[0].shape
        self.input_height, self.input_width = input_shape[0], input_shape[1]
        self._input_name = input_infos[0].name
        self._output_name = output_infos[0].name
        print(f"Model input: {self.input_width}x{self.input_height}")

        device = self._stack.enter_context(VDevice())
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        self._stack.enter_context(network_group.activate())
        self._pipeline = self._stack.enter_context(
            InferVStreams(network_group, input_vstreams_params, output_vstreams_params)
        )

        self._hailo_mon = monitor.StatsMonitor()
        self.max_tiles = self._batch_size - 1  # one slot reserved for full-frame
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def infer(self, images: list) -> list:
        """Run inference on a list of images.

        Parameters
        ----------
        images : list of np.ndarray
            Mix of tile crops (already input_height×input_width) and a full frame.

        Returns
        -------
        list of list
            One detection list per image; each detection is [x1, y1, x2, y2, conf, class_id].
        """
        n_real = len(images)
        tile_scale = (1.0, 0, 0, self.input_width, self.input_height)

        batch_imgs = []
        scale_infos = []
        for img in images:
            h, w = img.shape[:2]
            if h == self.input_height and w == self.input_width:
                batch_imgs.append(img)
                scale_infos.append(tile_scale)
            else:
                processed, scale_info = preprocess(img, self.input_height, self.input_width)
                batch_imgs.append(processed)
                scale_infos.append(scale_info)

        # Pad batch to fixed size
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        batch_imgs.extend([dummy] * (self._batch_size - len(batch_imgs)))

        outputs = self._pipeline.infer({self._input_name: np.stack(batch_imgs)})
        out_list = outputs[self._output_name]

        results = []
        for i in range(n_real):
            out = {self._output_name: [out_list[i]]}
            dets = postprocess(out, scale_infos[i], self._conf_thresh)
            results.append(dets)
        return results

    def get_hw_stats(self) -> dict:
        return self._hailo_mon.get()

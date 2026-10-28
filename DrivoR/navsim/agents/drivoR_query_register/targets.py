"""Paired future RGB without the diffusion variant's extra human noise anchor."""

from navsim.agents.drivoR_action_dit_register.targets import (
    FutureRegisterTargetBuilder, _human_future_targets,
)


class QueryFutureRegisterTargetBuilder(FutureRegisterTargetBuilder):
    def get_unique_name(self):
        width, height = self.image_size
        return f'drivor_query_future_rgb_v1_k{self.max_gt}_{width}x{height}'

    def compute_targets(self, scene):
        token = scene.scene_metadata.initial_token
        human = scene.get_future_trajectory(num_trajectory_frames=8).poses
        result = self.load(token, human)
        if bool(result['multi_gt_fallback']):
            # A missing export falls back to the actual human trajectory, so
            # pair that row with its actual future frames as well.
            logged = _human_future_targets(scene, self.sensor_blobs_path, self.image_size)
            result['future_register_images'][0] = logged['human_future_register_images']
            result['future_register_valid'][0] = logged['human_future_register_valid']
        return result

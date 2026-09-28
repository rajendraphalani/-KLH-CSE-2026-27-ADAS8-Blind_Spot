# Project Success Checklist

## Status Review: 2026-09-17

The runnable camera baseline is complete and the processed dataset currently contains
5,352 images and 5,355 labels. The three extra labels indicate orphan annotations;
the source annotation semantics, recording boundaries, label accuracy, and scene
conditions still require inspection of the original dataset. No radar recordings,
calibration, vehicle geometry, or hardware are present in this repository, so the
radar/system items below cannot be completed or validated here.

The remaining data/model items are intentionally left open until their required
evidence is available. Existing training output alone is not sufficient to claim
grouped-split validity, threshold tuning, model comparison, or reproducibility
checksums.

## Completed: Reproducible Camera Baseline

- [x] Prepare deterministic train/validation/test splits.
- [x] Normalize supported YOLO labels to the configured single class.
- [x] Reject malformed or out-of-range labels.
- [x] Report missing labels and orphan labels.
- [x] Remove stale generated split files on re-run.
- [x] Write a split manifest with source, seed, fractions, and counts.
- [x] Train with configurable model, epochs, image size, batch size, and device.
- [x] Evaluate on validation or held-out test data.
- [x] Export evaluation metrics as JSON for experiment tracking.
- [x] Run inference with configurable confidence, image size, and device.
- [x] Add regression tests for data preparation and CLI behavior.
- [x] Record a held-out benchmark for `bsd-colab-2`.

## Next: Data and Model Quality

- [ ] Confirm the meaning of every source annotation class before collapsing classes.
- [ ] Inspect label overlays and correct inaccurate or missing boxes.
- [ ] Replace frame-level random splitting with recording/scene-grouped splitting.
- [ ] Add difficult examples for small, occluded, low-light, and multi-object scenes.
- [ ] Compare `yolo11n`, `yolo11s`, image sizes, and training durations on validation data.
- [ ] Tune the confidence threshold for high recall without using the test set.
- [ ] Publish model, dataset, environment, and split-manifest checksums.
- [ ] Pin the Python and ML dependency versions used for the benchmark.

## Blocked: Radar and Vehicle System

These items require synchronized radar recordings, calibration data, vehicle geometry,
and hardware that are not present in this repository.

- [ ] Implement ADC ingestion, windowing, range/Doppler FFT, and range-Doppler maps.
- [ ] Implement radar peak detection, angle estimation, and vehicle-frame coordinates.
- [ ] Implement multi-target tracking and track continuity metrics.
- [ ] Define left/right blind-spot zones, persistence, hysteresis, and warning states.
- [ ] Add time-to-collision and threat-priority logic.
- [ ] Add synchronized camera-radar fusion.
- [ ] Validate range, velocity, angle, latency, and false-warning rate.
- [ ] Integrate CAN or another vehicle warning interface.
- [ ] Complete hardware-in-the-loop, road, weather, and failure-mode testing.
- [ ] Complete safety review before any real driving use.
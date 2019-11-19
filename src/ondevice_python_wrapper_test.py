"""Tests the python3.7 wrapper for ondevice inferencing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from absl.testing import absltest
from src import automl_ondevice


class OndevicePythonWrapperTest(absltest.TestCase):

  def test_full_pipeline(self):
    config = automl_ondevice.ObjectTrackingConfig(score_threshold=0.2)

    tracker = automl_ondevice.ObjectTrackingInference.TFLiteModel(
        'data/traffic_model.tflite',
        'data/traffic_label_map.pbtxt',
        config)

    input_size = tracker.getInputSize()
    self.assertEqual(input_size.width, 256)
    self.assertEqual(input_size.height, 256)

    image = Image.open(
        'data/traffic_frames/0001.bmp'
    ).convert('RGB').resize((input_size.width, input_size.height))

    # Tracker only appends to 'out', so it must be blank.
    out = []

    self.assertTrue(tracker.run(1, image, out))
    self.assertNotEmpty(out)
    self.assertLen(out, 5)

    for annotation in out:
      self.assertGreaterEqual(annotation.confidence_score, 0.2)
      print('{}: {} [{}, {}, {}, {}]'.format(annotation.class_name,
                                             annotation.confidence_score,
                                             annotation.bbox.top,
                                             annotation.bbox.left,
                                             annotation.bbox.bottom,
                                             annotation.bbox.right))


if __name__ == '__main__':
  absltest.main()

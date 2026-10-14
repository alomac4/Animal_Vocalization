"""Known synthetic geometry, conversion, invalid input and integration boundaries."""
import math
import unittest

from backend.distance import DistancePreset, format_distance
from backend.worker import detection_label


class DistanceTests(unittest.TestCase):
    def test_centered_pinhole_and_feet_conversion(self):
        value = DistancePreset(36, .5).estimate((450,150,550,250),1000,400)
        self.assertAlmostEqual(value['metres'],5)
        self.assertAlmostEqual(value['feet'],5/.3048)
        self.assertAlmostEqual(value['axial_metres'],5)
        self.assertEqual(format_distance(value),'est. 5.0 m / 16.4 ft')

    def test_off_axis_range_greater_than_axial_depth(self):
        value = DistancePreset(36,.5).estimate((650,150,750,250),1000,400)
        self.assertAlmostEqual(value['metres'],5*math.sqrt(1+.2**2))
        self.assertGreater(value['metres'],value['axial_metres'])

    def test_resolution_orientation_size_and_focal_scaling(self):
        preset=DistancePreset(36,.5)
        original=preset.estimate((450,150,550,250),1000,400)
        resized=preset.estimate((900,300,1100,500),2000,800)
        self.assertAlmostEqual(original['metres'],resized['metres'])
        portrait=preset.estimate((150,450,250,550),400,1000)
        self.assertAlmostEqual(original['metres'],portrait['metres'])
        self.assertAlmostEqual(DistancePreset(72,.5).estimate((450,150,550,250),1000,400)['metres'],10)
        self.assertAlmostEqual(DistancePreset(36,1).estimate((450,150,550,250),1000,400)['metres'],10)

    def test_clipped_tiny_invalid_and_nonfinite_boxes_are_unavailable(self):
        preset=DistancePreset()
        for box in [(0,10,50,100),(10,0,50,100),(10,10,1000,100),(10,10,50,400),
                    (10,10,50,20),(10,10,10,30),(50,10,10,30),(-10,10,30,50),
                    (10,10,float('nan'),100),(10,10,float('inf'),100),(10,10,100)]:
            self.assertIsNone(preset.estimate(box,1000,400),box)
        self.assertIsNone(preset.estimate((10,10,30,50),10,100))
        self.assertEqual(format_distance(None),'distance unavailable')

    def test_preset_validation_and_explicit_assumptions(self):
        for focal,extent in [(0,.65),(201,.65),(26,0),(26,3),(float('nan'),.65),(26,float('inf'))]:
            with self.assertRaises(ValueError): DistancePreset(focal,extent)
        metadata=DistancePreset().metadata()
        self.assertTrue(metadata['approximate'])
        self.assertEqual(metadata['focal_equivalent_mm'],26)
        self.assertEqual(metadata['dog_extent_m'],.65)
        self.assertIn('UNCALIBRATED',DistancePreset().notice())

    def test_video_label_does_not_leak_breed(self):
        detection=dict(track_id=2,breed={'name':'beagle'},distance={'metres':5,'feet':5/.3048})
        self.assertEqual(detection_label(detection),'Dog #2 | est. 5.0 m / 16.4 ft')
        detection['distance']=None
        self.assertEqual(detection_label(detection),'Dog #2 | distance unavailable')
        del detection['distance']
        self.assertEqual(detection_label(detection),'Dog #2')


if __name__=='__main__': unittest.main()
"""Depth sampling without downloaded weights/GPU; protocol error handling."""
import importlib.util
import queue
import unittest

from backend.depth_distance import DepthProClient, metadata


class DepthProtocolTests(unittest.TestCase):
    def test_metadata_does_not_use_assumed_height_or_lens(self):
        value=metadata()
        self.assertEqual(value['method'],'apple_depth_pro')
        self.assertTrue(value['approximate'])
        self.assertNotIn('dog_extent_m',value)
        self.assertNotIn('focal_equivalent_mm',value)

    def test_unavailable_installation_errors_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'installation missing'):
            DepthProClient('nonexistent-python.exe','nonexistent-model.pt')

    def test_service_failure_exit_and_timeout_are_explicit(self):
        client=object.__new__(DepthProClient)
        client.events=queue.Queue(); client.timeout=.001
        for value,message in [({'error':'CUDA out of memory'},'CUDA out of memory'),(None,'exited')]:
            client.events.put(value)
            with self.assertRaisesRegex(RuntimeError,message): client._receive()
        with self.assertRaisesRegex(RuntimeError,'timed out'): client._receive()


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Run project .venv for numpy sampling checks')
class DepthSamplingTests(unittest.TestCase):
    def test_central_median_rejects_box_background_edges(self):
        import numpy as np
        from backend.depth_distance import sample_box_depth
        depth=np.full((100,100),20,dtype=np.float32)
        depth[35:65,35:65]=3
        value=sample_box_depth(depth,(10,10,90,90))
        self.assertAlmostEqual(value['metres'],3)
        self.assertAlmostEqual(value['feet'],3/.3048)

    def test_invalid_mixed_tiny_and_outside_boxes_unavailable(self):
        import numpy as np
        from backend.depth_distance import sample_box_depth
        depth=np.full((100,100),3,dtype=np.float32)
        for box in [(1,1,4,4),(-1,0,50,50),(10,10,101,90),(50,10,10,30),(10,10,float('nan'),90)]:
            self.assertIsNone(sample_box_depth(depth,box))
        self.assertIsNone(sample_box_depth(np.full((100,100),np.nan),(10,10,90,90)))
        mixed=depth.copy(); mixed[:,50:]=20
        self.assertIsNone(sample_box_depth(mixed,(10,10,90,90)))
        self.assertIsNone(sample_box_depth(np.ones((10,10,3)),(0,0,10,10)))

    def test_partial_invalid_pixels_and_clipped_dog_can_still_have_depth(self):
        import numpy as np
        from backend.depth_distance import sample_box_depth
        depth=np.full((100,100),4,dtype=np.float32)
        depth[35:38,35:38]=np.nan
        value=sample_box_depth(depth,(0,0,100,100))
        self.assertEqual(value['metres'],4)
        self.assertEqual(value['axial_metres'],4)


if __name__=='__main__': unittest.main()
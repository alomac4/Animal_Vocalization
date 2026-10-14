"""Synthetic visual motion gaps: no fabricated detector observations/votes."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy','cv2','ultralytics','lap'))


@unittest.skipUnless(AVAILABLE, 'Run in project .venv with OpenCV/Ultralytics')
class GapMotionTests(unittest.TestCase):
    def scene(self, x, y=60):
        import numpy as np
        frame = np.full((220,360,3),20,dtype=np.uint8)
        texture = np.random.default_rng(12).integers(40,240,(60,70,3),dtype=np.uint8)
        frame[y:y+60,x:x+70] = texture
        return frame, (x,y,x+70,y+60)

    def test_follows_visible_pixels_across_missing_detections_then_expires(self):
        from backend.gap_motion import GapMotion
        motion=GapMotion(3)
        frame,box=self.scene(30)
        self.assertEqual(motion.advance(frame,0),{})
        motion.anchor(1,box,0)
        for number in range(1,4):
            frame,_=self.scene(30+number*5)
            hints=motion.advance(frame,number)
            self.assertIn(1,hints)
            self.assertAlmostEqual(hints[1][0],30+number*5,delta=1)
        self.assertEqual(motion.advance(self.scene(50)[0],4),{})

    def test_blank_crop_cut_missing_frame_and_order_do_not_produce_hints(self):
        import numpy as np
        from backend.gap_motion import GapMotion
        motion=GapMotion(10)
        frame,box=self.scene(30)
        motion.advance(frame,0); motion.anchor(1,box,0)
        self.assertEqual(motion.advance(np.full(frame.shape,250,dtype=np.uint8),1),{})
        self.assertEqual(motion.scene_cuts,1)
        self.assertEqual(motion.advance(None,2),{})
        with self.assertRaises(ValueError): motion.advance(frame,4)
        motion=GapMotion(10)
        blank=np.zeros(frame.shape,dtype=np.uint8)
        motion.advance(blank,0); motion.anchor(1,box,0)
        self.assertEqual(motion.advance(blank,1),{})

    def test_flow_keeps_identity_during_gap_but_adds_no_votes_or_boxes(self):
        from backend.temporal_tracking import TemporalDogTracker
        tracker=TemporalDogTracker(30,'ocsort_flow')
        previous_id=None
        for number in range(12):
            frame,box=self.scene(30+number*4)
            item=dict(box=box,confidence=.9,appearance=(1.,0.),breed=dict(name='beagle',confidence=.8))
            result=tracker.update([item],number,frame)
            previous_id=result[0]['track_id']
        track=tracker.active[previous_id]
        observed,votes=track.observations,sum(track.votes.values())
        for number in range(12,17):
            self.assertEqual(tracker.update([],number,self.scene(30+number*4)[0]),[])
        self.assertEqual(track.observations,observed)
        self.assertEqual(sum(track.votes.values()),votes)
        frame,box=self.scene(30+17*4)
        found=tracker.update([dict(box=box,confidence=.9,breed=dict(name='pug',confidence=.8))],17,frame)
        self.assertEqual(found[0]['track_id'],previous_id)
        self.assertGreater(tracker.backend.prediction_updates,0)

    def test_multiple_dogs_have_separate_visual_motion_and_ids(self):
        import numpy as np
        from backend.gap_motion import GapMotion
        motion=GapMotion(5)
        first,box=self.scene(30,30)
        texture=np.random.default_rng(50).integers(40,240,(50,50,3),dtype=np.uint8)
        first[150:200,220:270]=texture
        motion.advance(first,0); motion.anchor(1,box,0); motion.anchor(2,(220,150,270,200),0)
        second,_=self.scene(35,30); second[150:200,215:265]=texture
        hints=motion.advance(second,1)
        self.assertAlmostEqual(hints[1][0],35,delta=1)
        self.assertAlmostEqual(hints[2][0],215,delta=1)

    def test_complete_occlusion_rejects_features_instead_of_creating_observations(self):
        import numpy as np
        from backend.gap_motion import GapMotion
        motion=GapMotion(10)
        frame,box=self.scene(30)
        motion.advance(frame,0); motion.anchor(1,box,0)
        blank=np.full(frame.shape,20,dtype=np.uint8)
        self.assertEqual(motion.advance(blank,1),{})
        self.assertNotIn(1,motion.states)

    def test_motion_is_video_local_and_invalid_bounds_fail(self):
        from backend.gap_motion import GapMotion
        frame,box=self.scene(30)
        first=GapMotion(3); first.advance(frame,0); first.anchor(1,box,0)
        second=GapMotion(3)
        self.assertEqual(second.advance(frame,0),{})
        self.assertEqual(second.states,{})
        with self.assertRaises(ValueError): GapMotion(0)


if __name__=='__main__': unittest.main()
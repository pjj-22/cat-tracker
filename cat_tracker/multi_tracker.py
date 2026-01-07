"""
Multi-object tracker that manages multiple cat tracks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracker import Track
from .utils import iou, euclidean_distance


class MultiTracker:
    """
    Manages multiple tracks and performs data association.
    """
    
    def __init__(self, max_missed=10, min_hits=3, iou_threshold=0.3):
        """
        Initialize multi-tracker.
        
        Args:
            max_missed: Maximum frames a track can be missed before deletion
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching detection to track
        """
        self.tracks = []
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections, each is dict with 'box' and 'confidence'
            
        Returns:
            List of confirmed tracks
        """
        for track in self.tracks:
            track.predict()
        
        if len(detections) > 0 and len(self.tracks) > 0:
            matches, unmatched_dets, unmatched_tracks = self._match(detections)
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
        
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx]['box'],
                detections[det_idx]['confidence']
            )
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                detections[det_idx]['box'],
                detections[det_idx]['confidence']
            )
            self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.should_delete(self.max_missed)]
        
        # Return only confirmed tracks
        return [t for t in self.tracks if t.is_confirmed()]
    
    def _match(self, detections):
        """
        Match detections to existing tracks using Hungarian algorithm.
        
        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # Use IoU as similarity measure (convert to cost)
                iou_score = iou(track.predicted_bbox, det['box'])
                cost_matrix[i, j] = 1 - iou_score
        
        # Solve assignment problem
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out poor matches
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < (1 - self.iou_threshold):
                matches.append((track_idx, det_idx))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_dets, unmatched_tracks
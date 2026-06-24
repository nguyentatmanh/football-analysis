import numpy as np
from typing import List, Dict, Any
from football_ai.tracking.track_state import TrackState

class PossessionAnalyzer:
    """
    Tracks team possession percentages by determining spatial proximity 
    between the ball and nearest team players.
    """
    def __init__(self, grab_radius_m: float = 2.5, fps: float = 30.0, max_retention_seconds: float = 3.0):
        self.grab_radius_m = grab_radius_m
        self.fps = fps
        self.max_retention_frames = int(max_retention_seconds * fps)
        
        # State tracking for sticky possession (retention during passes and dropouts)
        self.last_possessor_team = None
        self.frames_since_last_touch = 0
        
        # Possession history timeline logging team_id per frame index
        self.timeline: Dict[int, int | None] = {}

    def process_frame(self, tracks: List[TrackState], frame_index: int):
        """
        Resolves the active possessor for the current frame by evaluating ball proximity,
        with retention logic for passes and minor tracking dropouts.
        """
        # 1. Find the ball
        ball = next((t for t in tracks if t.role == "ball" and t.pitch_xy is not None), None)
        
        if ball is None:
            # Handle temporary ball occlusion: retain possession for a limited time
            self.frames_since_last_touch += 1
            if self.last_possessor_team is not None and self.frames_since_last_touch <= self.max_retention_frames:
                self.timeline[frame_index] = self.last_possessor_team
            else:
                self.timeline[frame_index] = None
            return

        ball_xy = np.array(ball.pitch_xy)

        # 2. Gather human candidates with mapped coordinates
        candidates = [
            t for t in tracks 
            if t.role in ["player", "goalkeeper"] 
            and t.pitch_xy is not None 
            and t.team_id is not None
        ]

        if not candidates:
            self.frames_since_last_touch += 1
            if self.last_possessor_team is not None and self.frames_since_last_touch <= self.max_retention_frames:
                self.timeline[frame_index] = self.last_possessor_team
            else:
                self.timeline[frame_index] = None
            return

        # 3. Calculate Euclidean distances from ball to all human candidates
        min_dist = float('inf')
        assigned_team = None

        for c in candidates:
            dist = float(np.linalg.norm(np.array(c.pitch_xy) - ball_xy))
            if dist < min_dist:
                min_dist = dist
                assigned_team = c.team_id

        # 4. Enforce grab radius bounds with sticky possession fallback
        if min_dist <= self.grab_radius_m:
            self.timeline[frame_index] = assigned_team
            self.last_possessor_team = assigned_team
            self.frames_since_last_touch = 0
        else:
            # Ball is in transit (pass or clearance)
            self.frames_since_last_touch += 1
            if self.last_possessor_team is not None and self.frames_since_last_touch <= self.max_retention_frames:
                self.timeline[frame_index] = self.last_possessor_team
            else:
                self.timeline[frame_index] = None

    def compute_stats(self) -> Dict[str, Any]:
        """
        Aggregates timeline values to produce percentage distribution blocks.
        """
        frames = list(self.timeline.values())
        total_evaluations = len(frames)
        
        if total_evaluations == 0:
            return {
                "team_0_percent": 0.0,
                "team_1_percent": 0.0,
                "contested_or_none_percent": 100.0,
                "evaluated_frames": 0
            }

        # Counts
        count_0 = frames.count(0)
        count_1 = frames.count(1)
        count_none = frames.count(None)

        # Resolve aggregate total in-play frames (excluding purely missing ones)
        # We present percentage of TOTAL evaluation period for accuracy
        p0 = (count_0 / total_evaluations) * 100.0
        p1 = (count_1 / total_evaluations) * 100.0
        p_none = (count_none / total_evaluations) * 100.0

        # Also calculate normalized split between teams ONLY (standard broadcast format)
        team_total = count_0 + count_1
        if team_total > 0:
            split_0 = (count_0 / team_total) * 100.0
            split_1 = (count_1 / team_total) * 100.0
        else:
            split_0 = 50.0
            split_1 = 50.0

        return {
            "evaluated_frames": total_evaluations,
            "raw_timeline": [
                {"team_0_frames": count_0},
                {"team_1_frames": count_1},
                {"none_frames": count_none}
            ],
            "possession_distribution": {
                "team_a_percent": round(p0, 2),
                "team_b_percent": round(p1, 2),
                "none_percent": round(p_none, 2)
            },
            "broadcast_split": {
                "team_a_ratio": round(split_0, 1),
                "team_b_ratio": round(split_1, 1)
            }
        }

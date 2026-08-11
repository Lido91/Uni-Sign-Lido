"""COCO-WholeBody (RTMPose 133-point) index layout used by SignSMD-2D.

All indices are 0-based raw RTMPose indices, matching datasets.load_part_kp.
"""

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

# 68 iBUG face landmarks occupy raw indices 23..90.
FACE_START = 23
FACE_ANCHOR = 53          # iBUG point 30, nose tip (Uni-Sign face anchor)
FACE_BROWS = list(range(40, 50))        # iBUG 17-26, both eyebrows
FACE_RIGHT_EYE = list(range(59, 65))    # iBUG 36-41 (signer's right eye)
FACE_LEFT_EYE = list(range(65, 71))     # iBUG 42-47 (signer's left eye)
FACE_MOUTH_OUTER = list(range(71, 83))  # iBUG 48-59
FACE_MOUTH_INNER = list(range(83, 91))  # iBUG 60-67

# 21-point hands: offset 0 wrist, then 4 points per finger (thumb: CMC, MCP,
# IP, tip; others: MCP, PIP, DIP, tip).
LEFT_HAND_START = 91
RIGHT_HAND_START = 112
HAND_POINTS = 21
FINGER_OFFSETS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
MCP_OFFSETS = (5, 9, 13, 17)  # index/middle/ring/pinky MCP (palm scale/line)

SIDES = ("left", "right")
HAND_START = {"left": LEFT_HAND_START, "right": RIGHT_HAND_START}
BODY_WRIST = {"left": LEFT_WRIST, "right": RIGHT_WRIST}
BODY_ELBOW = {"left": LEFT_ELBOW, "right": RIGHT_ELBOW}
BODY_SHOULDER = {"left": LEFT_SHOULDER, "right": RIGHT_SHOULDER}

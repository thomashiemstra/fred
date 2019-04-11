import json

import jsonpickle
import numpy as np

from src.kinematics.kinematics_utils import Pose
from src.utils.movement import SplineMovement

arr = np.array([1.123, 1.123543], dtype=np.float64)
print(arr[0])
pose1 = Pose(1, arr[0], arr[0], alpha=arr[1], beta=arr[1], gamma=arr[1])
pose2 = Pose(2, arr[0], arr[0], alpha=arr[1], beta=arr[1], gamma=arr[1])

poses = [pose1, pose2]

test = SplineMovement(None, poses, 2)


json_string = jsonpickle.encode(test)

print(json.dumps(json.loads(json_string), indent=4))

with open('test.json', 'w') as outfile:
    json.dump(json.loads(json_string), outfile, indent=4)

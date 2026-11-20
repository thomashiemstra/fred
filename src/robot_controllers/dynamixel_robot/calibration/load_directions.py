"""
The labels of the direction column of the csv of record_static_load.py and record_load_sweep.py. The recorders measure
every pose from both directions and fit_static_load.py takes the mean of the two as the gravity load; half the
difference is the friction, which is not compensated. The labels live in a module of their own because the fit is run
offline and must not import the recorders, which pull in the robot and its hardware libraries.
"""

UP = 'up'  # the joint came from below, its angle was rising
DOWN = 'down'  # the joint came from above, its angle was falling

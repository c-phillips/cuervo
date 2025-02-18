import time
import logging
try:
    from rich.logging import RichHandler
    logging.basicConfig(handlers=[RichHandler()])
except ImportError:
    logging.basicConfig()

from dataclasses import dataclass

import numpy as np
import cuervo as cvo


# A "Resource" can be any object
class Radar:
    thresholds = [0.2, 0.5]
    def take_reading(self, velocities):
        too_fast = velocities > self.thresholds[1]
        too_slow = velocities < self.thresholds[0]
        speeding = np.argwhere(too_fast)
        safe = np.argwhere(~too_fast & ~too_slow)
        slow = np.argwhere(too_slow)
        return speeding, safe, slow

# Another "Resource" example is a dataclass
@dataclass
class SpeedTracker:
    speeding: set[int]
    safe: set[int]
    slow: set[int]


class State(cvo.Component):
    shape = (4,)


class TrafficMonitor(cvo.System):
    def process(
        self,
        x: State,
    ):
        radar = self.ecs.resources[Radar]
        tracker = self.ecs.resources[SpeedTracker]
        readings = radar.take_reading(x[:,2:])
        tracker.speeding = readings[0]
        tracker.safe = readings[1]
        tracker.slow = readings[2]


class MovementSystem(cvo.System):
    def process(
        self,
        x: State,
        dt: float,
    ):
        x[:,:2] += x[:,2:] * dt
        self.update_component(State, x)


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ecs = cvo.ECS()

    ecs.add_resource(SpeedTracker(set(), set(), set()))
    ecs.add_resource(Radar())

    ecs.add_system(MovementSystem, priority=1)
    ecs.add_system(TrafficMonitor, priority=2)
    ecs.add_entity(
        systems=[MovementSystem, TrafficMonitor],
        initial_values={
            State: np.random.normal(0,1,(10000,4))  # random values
        },
        batch=True
    )

    logger.info("Starting loop")
    start_time = time.process_time()
    dt = 0.1
    t = 0.0
    while t < 5.0:
        ecs.process(
            t=t,
            dt=dt
        )
        t += dt
    total_time = time.process_time() - start_time
    logger.info(f"Done! Took {total_time:0.6}s")
    logger.info(f"There were {len(ecs.resources[SpeedTracker].speeding)} speeding entities")

if __name__ == "__main__":
    main()

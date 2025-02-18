import time
import logging
try:
    from rich.logging import RichHandler
    logging.basicConfig(handlers=[RichHandler()])
except ImportError:
    logging.basicConfig()

import numpy as np
import cuervo as cvo

# Use cuervo.BufferComponent instead of cuervo.Component when you are
# frequently adding/removing entities with this component
class State(cvo.BufferComponent):
    shape = (4,)


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

    # Have to manually instantiate the buffer to give it a default size using
    # cuervo's special annotated class notation
    ecs.add_component(State[10000]())

    ecs.add_system(MovementSystem)
    ecs.add_entity(
        systems=[MovementSystem],
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

if __name__ == "__main__":
    main()

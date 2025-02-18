import time
import logging
try:
    from rich.logging import RichHandler
    logging.basicConfig(handlers=[RichHandler()])
except ImportError:
    logging.basicConfig()

import numpy as np
import cuervo as cvo


class SetupSystem(cvo.System):  
    setup = True
    def process(self):
        self.logger.info("Runs just once!")

    @cvo.ECS.connect("test_event")
    def test_event_handler(self):
        self.logger.info("events still trigger though")


class State(cvo.Component):
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

    ecs.add_system(MovementSystem)
    ecs.add_system(SetupSystem)

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

    ecs.trigger("test_event")

    logger.info(f"Done! Took {total_time:0.6}s")

if __name__ == "__main__":
    main()

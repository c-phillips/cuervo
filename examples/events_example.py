import time
import logging
try:
    from rich.logging import RichHandler
    logging.basicConfig(handlers=[RichHandler()])
except ImportError:
    logging.basicConfig()

import numpy as np
import cuervo as cvo


class EventCaller:
    def __init__(self, ecs: cvo.ECS):
        self.ecs = ecs

    def trigger(self, event: str, *args, **kwargs):
        self.ecs.trigger(event, *args, **kwargs)


class DummySystem(cvo.System):
    def process(self):
        self.logger.info("dummy process...")
    
    @cvo.ECS.connect("test_event")
    def test_event_handler(self, ecs: cvo.ECS, **kwargs):
        self.logger.info(kwargs)


class AnotherDummySystem(cvo.System):
    def process(self):
        self.logger.info("another dummy process...")
    
    @cvo.ECS.connect("test_event")
    def test_event_handler(self, option, **event):
        self.logger.info("within another dummy handler")


def test_event_handler(ecs: cvo.ECS, **kwargs):
    logger = logging.getLogger("example.test_handler")
    entities = ecs.components[State].entities
    logger.info(f"Found {len(entities)} entities with a State component")


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

    eventer = EventCaller(ecs)
    ecs.connect("test_event", test_event_handler)

    ecs.add_system(MovementSystem)
    ecs.add_system(DummySystem)
    ecs.add_system(AnotherDummySystem)

    ecs.add_entity(
        systems=[MovementSystem],
        initial_values={
            State: np.random.normal(0,1,(10000,4))  # random values
        },
        batch=True
    )
    logger.info("Starting loop")

    eventer.trigger("test_event", option="an option")
    eventer.trigger("test_event", option="deferred event", defer=True)

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

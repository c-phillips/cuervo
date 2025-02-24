import logging

import numpy as np
import cuervo as cvo


def test_component():
    logger = logging.getLogger("test")
    
    # Component for testing
    class Position(cvo.Component):
        shape = (3,)  # x,y,z coordinates

    ecs = cvo.ECS()  # and our ECS

    # We need some initial data for creating our entities
    N = 20
    positions = np.hstack([
        np.random.uniform(-10, 10, (N, 2)).astype(np.float32),
        np.zeros((N,1), dtype=np.float32)+0.5
    ])

    # Use our initial data to batch-create some entities
    ecs.add_entity(
        initial_values={
            Position: positions
        },
        batch = True
    )

    # Check that the values on the component itself are correct
    vals = ecs.get_values(Position)
    cmp  = ecs.components[Position]
    assert np.all(vals == positions),\
        "Initialized values aren't correct!"
    
    # Validate that removing a single entity works
    ecs.delete_entity(2)
    vals = ecs.get_values(Position)
    logger.debug(vals)
    assert len(vals) == N-1,\
        "Returned incorrect values"

    # Validate the removal of a number of entities
    ecs.delete_entity([5,10,15])
    vals = ecs.get_values(Position)
    logger.debug(vals)
    assert len(vals) == N-4,\
        "Remove multiple eids: Returned incorrect values"

    # Validate that adding an entity will fill recently vacated slots first
    ecs.add_entity(
        initial_values = {
            Position: np.array([9, 9, 9], dtype=np.float32)
        }
    )
    vals = ecs.get_values(Position)
    logger.debug(vals)
    assert len(vals) == N-3,\
        "Add single back: returned incorrect values"

    # Validate that adding multiple entities will fill recently vacated slots
    ecs.add_entity(
        initial_values = {
            Position: np.array([
                [10, 10, 10],
                [11, 11, 11]
            ], dtype=np.float32)
        },
        batch=True
    )
    vals = ecs.get_values(Position)
    logger.debug(vals)
    assert len(vals) == N-1,\
        "Add mult back: returned incorrect values"

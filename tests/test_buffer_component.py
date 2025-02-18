import logging

import numpy as np
import cuervo as cvo


def test_buffer_component():
    logger = logging.getLogger("test")
    
    # Component for testing
    class Position(cvo.BufferComponent):
        shape = (3,)  # x,y,z coordinates

    # Start by defining a length-50 buffered component
    comp_type = Position[50]
    ecs = cvo.ECS()  # and our ECS
    ecs.add_component(comp_type())

    # We need some initial data for creating our entities
    N = 20
    positions = np.hstack([
        np.random.uniform(-10, 10, (N, 2)).astype(np.float32),
        np.zeros((N,1), dtype=np.float32)+0.5
    ])

    # Use our initial data to batch-create some entities
    ecs.new_entity(
        initial_values={
            Position: positions
        },
        batch = True
    )

    # Check that the values on the component itself are correct
    vals = ecs.get_values(Position)
    cmp  = ecs.components[Position]
    assert np.all(cmp.values[cmp.active] == vals),\
        "Selected values do not equal the active values"
    assert np.all(vals == positions),\
        "Initialized values aren't correct!"
    
    # Validate that removing a single entity works
    ecs.delete_entity(2)
    vals = ecs.get_values(Position)
    logger.debug(vals)
    logger.debug(cmp.active)
    assert len(vals) == N-1,\
        "Returned incorrect values"
    assert np.sum(cmp.active) == N-1,\
        "Component's active mask is the wrong length"
    assert not cmp.active[1],\
        "Component's active mask did not update correctly"

    # Validate the removal of a number of entities
    ecs.delete_entity([5,10,15])
    vals = ecs.get_values(Position)
    logger.debug(vals)
    logger.debug(cmp.active)
    assert len(vals) == N-4,\
        "Remove multiple eids: Returned incorrect values"
    assert np.sum(cmp.active) == N-4,\
        "Remove multiple eids: active mask is the wrong length"
    assert not (cmp.active[1] | cmp.active[4] | cmp.active[9] | cmp.active[14]),\
        "Remove multiple eids: active mask did not update correctly"

    # Validate that adding an entity will fill recently vacated slots first
    ecs.new_entity(
        initial_values = {
            Position: np.array([9, 9, 9], dtype=np.float32)
        }
    )
    vals = ecs.get_values(Position)
    logger.debug(vals)
    logger.debug(cmp.active)
    assert len(vals) == N-3,\
        "Add single back: returned incorrect values"
    assert np.sum(cmp.active) == N-3,\
        "Add single back: active mask wrong length"
    assert not (cmp.active[4] | cmp.active[9] | cmp.active[14]) and cmp.active[1],\
        "Add single back: active mask did not update correctly"

    # Validate that adding multiple entities will fill recently vacated slots
    ecs.new_entity(
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
    logger.debug(cmp.active)
    assert len(vals) == N-1,\
        "Add mult back: returned incorrect values"
    assert np.sum(cmp.active) == N-1,\
        "Add mult back: active mask wrong length"
    assert cmp.active[1] and cmp.active[4] and cmp.active[9] and (not cmp.active[14]),\
        "Add mult back: active mask did not update correctly"

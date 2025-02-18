from typing import Any, Optional, Union, Set, Callable, get_type_hints, Iterable
import inspect
from functools import partial
from dataclasses import dataclass
import logging

import numpy as np
import numpy.typing as npt

from .System import System
from .Component import Component


@dataclass
class Event:
    """A dataclass for holding metadata about an event"""
    name: str
    args: list[Any]
    kwargs: dict[str,Any]
    deferred: bool = False
    originator: Optional[type[System]] = None


class ECS:
    """Entity Component System

    Attributes
    ----------
    logger: logging.Logger
        logger instance for all your logging needs
    entities: Set[int]
        a set of integers representing entity ids
    systems: dict[type[System], System]
        a mapping between system types and an instance of that system
    execution_order: list[list[type[System]]]
        a list of lists of system types specifying the order in which systems
        are processed
    components: dict[type[Component], Component]
        a mapping between component types and the component instances
    deferred_events: list[Event]
        a list of [`Event`][raven.ecs.ECS.Event] objects which have been stored
        for evaluation after all systems have finished processing
    handlers: dict[str, list[Callable]]
        a mapping between event names and their handlers
    rng: np.random.Generator
        a numpy random number generator, seeded by the `seed` passed on init

    _wrapped_handlers: dict[type[System], dict[str, Callable]]
        a private mapping between systems and their mapped event handlers which
        were decorated with the `ECS.connect` function
    _max_eid: int
        a private integer which increments whenever a new entity is created
    """

    def make_connect_decorator():
        """A meta function which wraps an inner decorator for the purpose of
        creating a registry of wrapped functions.
        """
        event_handlers: dict[str, str] = {}
        def connect(event: str):
            def decorator(function):
                owner_cls_name, function_name = function.__qualname__.split('.')
                if owner_cls_name not in event_handlers:
                    event_handlers[owner_cls_name] = {}
                event_handlers[owner_cls_name][event] = function_name
                return function
            return decorator
        connect.event_handlers = event_handlers
        return connect
    connect = make_connect_decorator()

    def __init__(self, seed: int = None):
        """Initialize an ECS

        Parameters
        ----------
        seed: int
            a seed to use for a numpy random number generator
        """
        self.logger = logging.getLogger("ECS")
        self.entities: Set[int] = set()
        self._max_eid = 0

        self.systems: dict[type[System], System] = {}
        self.execution_order: list[list[type[System]]] = [[]]
        self.components: dict[type[Component], Component] = {}
        self.setup_systems: list[type[System]] = []

        self.rng = np.random.default_rng(seed=seed)

        self.deferred_events: list[Event] = []
        self.handlers: dict[str, list[Callable]] = {}

        # TODO: determine if this is the right way to do a class method registration decorator
        self._wrapped_handlers: dict[type[System], dict[str, Callable]] = self.connect.event_handlers
        def connect(event: str, handler: Callable):
            if event not in self.handlers:
                self.handlers[event] = []
            hints = get_type_hints(handler)
            if type(self) in hints.values():
                self.handlers[event].append(partial(handler, ecs=self))
            else:
                self.handlers[event].append(handler)
        self.connect = connect
    
    def trigger(
        self,
        event_name: str,
        *args,
        defer: bool = False,
        **kwargs
    ):
        """Trigger an event and pass appropriate arguments to registered event
        handlers.

        This method is the avenue through which [events](events.md) are
        propagated. The arguments provided on the event trigger are sent
        through to whatever handlers are registered.

        Parameters
        ----------
        event_name: str
            the name of the event to trigger
        defer: bool, default=False
            flag to defer the event until after all systems have processed
        *args
            positional arguments which, if present, are passed to the event
            handler
        **kwargs
            keyword arguments which, if present, are passed to the event
            handler
        """
        tb = inspect.getouterframes(inspect.currentframe(), 2)[1]
        event = Event(name=event_name, args=args, kwargs=kwargs, deferred=defer)
        if caller := tb.frame.f_locals.get("self", False):
            self.logger.debug(f"event <{event_name}> triggered by {type(caller).__name__}")
            kwargs["event_originator"] = type(caller).__name__
        
        if defer:
            self.deferred_events.append(event)
            return

        for handler in self.handlers.get(event_name, []):
            handler(*args, **kwargs)

    def get_values(
        self,
        component_cls: type[Component],
    ) -> Union[object, np.ndarray]:
        """Return the values for all entities with the provided component.

        Parameters
        ----------
        component_cls: type[Component]
            the component from which to retrieve entity values
        """
        component = self.components.get(component_cls, None)
        if component is None:
            raise KeyError(f"Could not find any registered component: {component_cls}!")
        return component.convert(component.get_values())
    
    def process(self,*args,**kwargs):
        """Calls each system in priority order and passes along any provided
        arguments. After each system finishes processing deferred
        [events](events.md) are triggered.
        """
        if self.setup_systems:
            while self.setup_systems:
                system = self.systems.pop(self.setup_systems.pop())
                system(*args, **kwargs)

        for priority_level in self.execution_order:
            for system_cls in priority_level:
                system = self.systems[system_cls]
                if len(system.registered_eids) > 0:
                    system(*args, **kwargs)
        
        if self.deferred_events:
            for event in self.deferred_events:
                for handler in self.handlers[event.name]:
                    handler(*event.args, **event.kwargs)
            self.deferred_events = []
    
    def add_component(
        self,
        component: Component,
        override: bool = False,
    ):
        """Adds a component instance to the managed available components.

        Parameters
        ----------
        component
            the `Component` instance to add
        override
            if the component already exists, replace it instead of throwing
        
        Raises
        ------
        KeyError
            if the component already exists and override is not `True`
        """
        T = type(component)
        if T in self.components and not override:
            raise KeyError(f"Component<{T.__name__}> has already been created!")
        self.components[T] = component

    def add_system(
        self,
        system_cls: type[System],
        *args,
        priority: int = 1,
        **kwargs
    ):
        """Adds a system (and its components) to the ECS.

        Whenever a new system is added, the required components specified by
        that system are also instantiated if they haven't been already.
        Additionally, any wrapped trigger handlers are registered with the ECS.

        Parameters
        ----------
        system_cls: type[System]
            the system to register; note: this is a *type* not an instance
        priority: int, default=1
            the execution priority order where an increasing number indicates
            lower priority, e.g. priority 0 is highest
        *args, **kwargs:
            passed along to the system initialization call
        """
        if system_cls in self.systems:
            raise KeyError(f"Already registered a system of type {system_cls}!")

        components = []
        for component_cls in system_cls.required_components:
            if component_cls not in self.components:
                self.components[component_cls] = component_cls()
            c = self.components[component_cls]
            components.append(c)

        self.systems[system_cls] = system_cls(
            components=components,
            ecs=self,
            config=kwargs
        )
        if event_methodname_map := self._wrapped_handlers.get(system_cls.__name__, False):
            for event, handler_name in event_methodname_map.items():
                handler = getattr(self.systems[system_cls], handler_name)
                self.connect(event, handler)

        if system_cls.setup:
            self.setup_systems.append(system_cls)
        else:
            while priority > (len(self.execution_order)-1):
                self.execution_order.extend([[]*(priority - len(self.execution_order) + 1)])
            self.execution_order[priority].append(system_cls)

    def deregister_entity(self, system_cls: type[System], eids: int | Iterable[int]):
        """Remove an entity from being processed by the provided system
        """
        if not hasattr(eids, "__len__"): eids = [eids]
        eids = set(eids)
        
        # TODO: Efficiently check if entity is being processed by ANY system
        #       and optionally delete the entity if so
        sys = self.systems.get(system_cls, None)
        if sys is not None:
            for eid in eids.intersection(sys.registered_eids):
                sys.remove_entity(eid)
        else:
            self.logger.warning(f"ECS.deregister_entity could not find system: {system_cls.__name__}")
    
    def delete_entity(self, eids: int | Iterable[int]):
        if not hasattr(eids, "__len__"): eids = [eids]
        eids = set(eids)

        # TODO: Find a more efficient way to do this that minimizes loops
        for component in self.components.values():
            for eid in eids.intersection(component.entities):
                component.remove_entity(eid)
        for system in self.systems.values():
            for eid in eids.intersection(system.registered_eids):
                system.remove_entity(eid)

    def add_entity(
        self,
        systems: list[type[System]] | None = None,
        initial_values: dict[type[Component], npt.ArrayLike] | None = None,
        batch: bool = False,
    ) -> int | list[int]:
        """Create and register a new entity within the ECS.

        This method will create a new `Entity` metadata class and add it to the
        ECS's internal entities map. Entities are represented as collections of
        components, and are registered for processing by systems.

        If any of the systems are not already initialized a `KeyError` will be
        thrown. However, if any components are not already initialized, they
        will be initialized before adding the new entities.

        Parameters
        ----------
        systems: list[type[Systems]]
            a list of the sytems for which the entity should be registered
        initial_values: dict[type[Component], npt.ArrayLike], default={}
            a mapping of components to their initial values
        batch: bool, default=False
            whether to add many insances of this new entity. If so, the first
            dimension of the component values will indicate the number of new
            entities to be added
        """
        if systems is None: systems = []
        if initial_values is None: initial_values = {}
        NUM_ENTITIES = 1 if not batch else initial_values[list(initial_values.keys())[0]].shape[0]

        # add our new entity to the entity registry
        eids = []
        for _ in range(NUM_ENTITIES):
            self._max_eid += 1
            self.entities.add(self._max_eid)
            eids.append(self._max_eid)
            
        # use the provided initial values to create the requisite components
        for component_cls, value in initial_values.items():
            if component_cls not in self.components:
                self.components[component_cls] = component_cls()
            self.components[component_cls].add_entity(eids if batch else eids[0], value, batch=batch)

        # finally, register this new entity with each of the requested systems
        for system_cls in systems:
            if system_cls not in self.systems:
                raise KeyError(f"Could not find any registered system of type {system_cls}!")
            system = self.systems[system_cls]                
            system.add_entity(
                eids if batch else eids[0],
                batch=batch
            )
        
        return eids[0] if len(eids) == 1 else eids

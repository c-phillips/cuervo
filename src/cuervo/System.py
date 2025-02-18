from typing import get_type_hints, Set, Optional, Union, Sequence, NamedTuple, TYPE_CHECKING
from operator import itemgetter
import logging

import numpy as np
import numpy.typing as npt

from .Component import Component
from .util import classproperty

if TYPE_CHECKING:
    from .ecs import ECS


class System:
    """The basic algorithmic component of an entity component system.

    Within an ECS, the `System` processes data stored in components and is
    generally responsible for any dynamic changes and updates to those values.
    Any system which inherits from the base `System` class must implement a
    `process` method that specifies the components it expects to receive in
    the form of type annotations. When called, the system will query the
    relevant `Component`s for entity values before passing it along to the 
    user-defined `process` implementation. 
    
    Attributes
    ----------
    logger: logging.Logger
        a logger for all your logging needs
    ecs: cuervo.ecs.ECS
        the ECS to which this system is registered
    registered_eids: Set[int]
        a set of registered unique entity ids
    eidx: dict[type[Component], npt.NDArray[np.uint32]]
        a mapping between component types and an array of entity indices
    components: list[Component]
        a list of `Component` instances
    cmap: dict[type[Component], Component]
        a mapping between `Component` types and their assigned instances

    required_components: Set[type[Component]]
        a set of the components required to fulfill the requirements of the
        system's `process` method's parameters
    
    setup: bool
        if `True` run the process method only once
    

    Examples
    --------
    Lets consider an example where we have a joint 2D position and velocity 
    component that we would like to update on each time-step:
    >>> class State(Component):
    ...     shape = (4,)

    >>> class StateUpdate(System):
    ...     def process(
    ...        self, 
    ...        states: State,
    ...        timedelta: float
    ...     ):
    ...         states[:,:2] += states[:,2:] * timedelta
    ...         self.update_component(State, states)
    """
    _call_args: dict[str, type] = None
    _call_arg_map: dict[type, str] = None
    _required_components: Set[type[Component]] = None

    setup: bool = False

    def __init__(
        self,
        components: list[Component],
        ecs: 'ECS' = None,
        config = None,
    ):
        self.logger = logging.getLogger(f"ECS.S.{type(self).__name__}")

        self.ecs = ecs

        self.registered_eids = set()
        self.eidx: dict[type[Component], npt.NDArray[np.uint32]] = {type(c):np.empty((2,0), dtype=np.uint32) for c in components}

        self.components = components
        self.cmap = {type(c): c for c in self.components}

        self._gathered = None
        self.logger.debug(f"finished init: {list(map(lambda c: type(c).__name__, components))}")

    @classproperty
    def required_components(cls) -> Set[type[Component]]:
        """A class property method that determines the set of required
        component types for the defined process method.

        Notes:
        ------
        This is much of the magic that makes the System class easy to use.
        We take full advantage of Python's type annotations to perform 
        runtime type reflection and generate the list of required types.
        This list is used when registering a system with the ECS and also for
        gathering together the values of the parameters for the process method.

        """
        if cls._required_components is None:
            hints = get_type_hints(cls.process)
            try:
                hints.pop('return')
            except Exception: pass

            cls._call_args = hints
            call_noncomp = []
            cls._call_arg_map = {}
            cls._required_components = set()

            for k,v in hints.items():
                if type(v) is type(Component) and issubclass(v, Component):
                    cls._call_arg_map[v] = k
                    cls._required_components.add(v)

                elif hasattr(v, "_fields"):
                    cls._call_arg_map[v] = k
                    
                    fields = get_type_hints(v)
                    for _,sv in fields.items():
                        cls._required_components.add(sv)

                else:
                    call_noncomp.append(k)

            if call_noncomp:
                # TODO: investigate performance of the length check
                def fn(_,kwargs):
                    items = itemgetter(*call_noncomp)(kwargs)
                    return dict(zip(call_noncomp, items if len(call_noncomp) > 1 else [items], strict=True))
                cls._convert_call_kwargs = fn
            else:
                cls._convert_call_kwargs = lambda _,__: {}

        return cls._required_components
    
    def _update_eidx(self):
        # TODO: Investigate performance when removing entities frequently
        for component_cls, comp in self.cmap.items():
            shared_entities = comp.entities.intersection(self.registered_eids)
            self.eidx[component_cls] = comp.eidx_mat(shared_entities) if shared_entities else np.array([], dtype=np.uint32)

    def add_entity(
        self,
        eid: Union[int, Sequence[int]],
        batch: bool = False,
    ):
        """Register an entity by ID with this system.

        Parameters
        ----------
        eid: Union[int, Sequence[int]]
            the entity ID(s) to register
        batch: bool, default=False
            a flag used to tell this method if there are multiple entities
        """
        # TODO: determine if we should just check the type of the passed eid
        #       parameter, or if we keep it as is for API consistency
        if not batch: eid = [eid]
        eid = set(eid)
        self.registered_eids.update(eid)
        self._update_eidx()

    def deregister_entity(self, eid: int):
        """Remove an entity ID from the registered set"""
        self.registered_eids.remove(eid)
        self._update_eidx()

    def _pull_component(
        self,
        component: Union[type[Component], NamedTuple],
        entities: Optional[Set[int]] = None
    ) -> Union[npt.NDArray, dict[str, npt.NDArray]]:
        """Recursively pull component values and aggregate them into the
        required structure for passing through to the `process` method

        Parameters
        ----------
        component: Union[type[Component], NamedTuple]
            the "component" to pull values for; if this parameter is a
            `NamedTuple`, we will instead recursively call this method on the
            fields of the compound component
        entities: Optional[Set[int]]
            a set of entities to pull. useful for compound components
        
        Returns
        -------
        Union[npt.NDArray, dict[str, npt.NDArray]]
            component values or dictionary of component values corresponding to
            the fields of a compound component
        """
        comp = self.cmap.get(component, None)
        if comp is None:
            entities = self.registered_eids if entities is None else entities
            for comp_cls in get_type_hints(component).values():
                entities = entities.intersection(self.cmap[comp_cls].entities)
            return {name: self._pull_component(comp_cls, entities) for name,comp_cls in get_type_hints(component).items()}

        eids = comp.entities.intersection(entities if entities is not None else self.registered_eids)
        return comp.convert(comp[eids])

    def _fill_args(self) -> dict[str, Union[npt.NDArray, dict[str, npt.NDArray]]]:
        """Helper method to pull component values"""
        return {
            name: self._pull_component(comp) for comp, name in self._call_arg_map.items()
        }

    def update_component(self, component: type[Component], new_values: object):
        """Update the values on a specified component.

        Parameters
        ----------
        component: type[Component]
            the component type to update; must be a component recognized by the
            system and specified in the process call parameters
        new_values: object
            the new value to set for the component
        """
        if hasattr(component, "_fields"):
            fields = get_type_hints(component)
            entities = self.registered_eids
            for comp_cls in fields.values():
                entities = entities.intersection(self.cmap[comp_cls].entities)

            for sk,sv in fields.items():
                eidx = self.eidx[sv][1,np.isin(self.eidx[sv][0], list(entities))]
                self.cmap[sv].set_values(sv.revert(new_values[sk]), eidx)
        else:
            self.cmap[component].set_values(component.revert(new_values), self.eidx[component][1])

    def process(
        self,
        *args,
        **kwargs
    ):
        """Core algorithmic method for processing and updating component data.

        Every system must implement its own `process` method.
        The type annotations of the method parameters are critical and define
        the required components for the system to query against.

        >[!warning] You will almost never want to call the process method
                    directly. Instead, use `my_system()` calling convention

        Raises
        ------
        NotImplementedError
            If the process method is not implemented on a child class
        
        Examples
        --------
        Lets look an example `System` that propagates entity positions using
        entity velocities given a time-delta:

        >>> class MySystem(System):
        ...     def process(
        ...        self, 
        ...        positions: Position, 
        ...        velocities: Velocity, 
        ...        timedelta: float
        ...     ):
        ...         positions += velocities * timedelta
        ...         self.update_component(Position, positions)

        Notice that there is no additional machinery required to query for
        component values. By annotating the parameters of the process method,
        the `System.required_components` classmethod will be able to infer the
        set of required components and pass them along to the ECS. The
        `System.__call__` method will also use the type annotations to aggregate
        the call parameters and pull entity values from the component instances.

        Additionally, notice that the system has an `update_component` step
        which will set new position values on the Position component. Again, all
        of the machinery to set entity indices and split values is already 
        implemented on the backend.
        """
        raise NotImplementedError("All systems must implement a `process` method!")

    def __call__(
        self,
        *args,
        **kwargs
    ) -> npt.NDArray:
        """Allows for the user to conveniently call the system on new data. 
        
        This method pulls and organizes the process method paramters alongside
        any additional parameters which are passed to this call directly.
        """
        call_kwargs = self._convert_call_kwargs(kwargs) | self._fill_args()
        return self.process(*args,**call_kwargs)

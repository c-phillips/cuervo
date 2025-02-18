from typing import Set, Union, Iterable, Optional
import logging

import numpy as np
import numpy.typing as npt


class Component:
    """The base component class for the entity component system.

    New components should inherit from this class and optionally implement
    a convert and/or revert method. These are called before/after processing
    /updating respectively.

    Generally, the underlying data is (and should be) stored as a numpy array.
    The first dimension of the data is the number of entities such that each
    entity's data is a row vector.

    Attributes
    ----------
    shape: Iterable[int]
        the shape of the underlying numpy array data
    dtype: np.dtype, default=np.float32
        the data type of the underlying numpy array data
    
    logger: logging.Logger
        a logger for all your logging needs
    entities: Set[int]
        a set of unique entity IDs corresponding to the entities with this
        component
    eid_to_idx: dict[int, int]
        a map between entity IDs and underlying data indices
    values: npt.NDArray
        the underlying numpy array data

    Examples
    --------
    A simple 3-dimensional position component would look like:
    >>> class Position(Component):
    ...     shape = (3,)

    We can also perform operations like reshaping or structured typing before
    the internal data is passed to a system or updated:
    >>> class StructuredType(Component):
    ...     shape = (5,)
    ...     repr_dtype = np.dtype([
    ...         ('a', 'f4', 3),
    ...         ('b', 'f4', 2)
    ...     ])
    ...     @classmethod
    ...     def convert(cls, value):
    ...         return value.view(cls.repr_dtype)

    Then, in a system which requires the `StructuredType` component will be
    passed the data in the converted format such that the entries can be
    accessed like `value['a']`. 
    """
    shape: Iterable[int]=()
    dtype: np.dtype = np.float32
    
    def __init__(self):
        self.logger = logging.getLogger(f"ECS.C.{type(self).__name__}")
        self.entities: Set[int] = set()
        self.eid_to_idx: dict[int, int] = {}
        self.values: npt.NDArray = np.empty((0,*self.shape), dtype=self.dtype)
        self.logger.debug("finished init")

    @classmethod
    def convert(cls, values: np.ndarray) -> object:
        return values
    
    @classmethod
    def revert(cls, values: object) -> np.ndarray:
        return values
    
    def get_values(self, eidx: Optional[Iterable[int]] = None) -> npt.NDArray:
        if eidx is None: return self.values
        return self.values[eidx]
    
    def set_values(self, values: npt.NDArray, eidx: Optional[Iterable[int]] = None):
        if eidx is None:
            self.values = values
        else:
            self.values[eidx] = values
    
    def eidx_mat(self, eids: Optional[Iterable[int]] = None) -> npt.NDArray[np.uint32]:
        """A convenience method that returns a matrix where the first column
        contains the entity ID and the second returns that entity's value matrix
        row index.

        Parameters
        ----------
        eids: Optional[Iterable[int]], default=None
            if `None` all entities are returned
        """
        if eids is None:
            return np.array([[eid,idx] for eid,idx in self.eid_to_idx.items()], dtype=np.uint32)
        return np.stack([np.array(list(eids), dtype=np.uint32), self.map_eids(eids)])

    def add_entity(
        self,
        eid: Union[int, Iterable[int]],
        value: npt.ArrayLike,
        batch: bool = False
    ):
        """Adds new entities to this component and sets the initial values.

        Parameters
        ----------
        eid: Union[int, Iterable[int]]
            the entity ID or Iterable of IDs to add
        value: npt.ArrayLike
            the values to add for the provided entity/entities
        batch: bool, default=False
            if this flag is set, the eid is assumed to be a Iterable of values
            and the first dimension of the `value` parameter is assumed to be
            the number of entities being added
            
        """
        if batch:
            self.entities.update(eid)
            self.values = np.append(self.values, value.astype(self.dtype), axis=0)
            self.eid_to_idx |= {e:self.values.shape[0]-(i+1) for i,e in enumerate(eid)}
        else:
            self.entities.add(eid)
            self.values = np.append(self.values, [value.astype(self.dtype)], axis=0)
            self.eid_to_idx[eid] = self.values.shape[0]-1
    
    def remove_entity(self, eid: int):
        """Removes an entity and its value from the component
        
        Parameters
        ----------
        eid: int
            the entity ID to remove
        """
        self.entities.remove(eid)
        idx = self.eid_to_idx.pop(eid)
        self.values = np.delete(self.values, idx, axis=0)

        # TODO: find a faster way to do this
        for other_eid, other_idx in self.eid_to_idx.items():
            if other_idx > idx:
                self.eid_to_idx[other_eid] = other_idx - 1

    def map_eids(self, eids: Iterable[int]) -> npt.NDArray[np.uint32]:
        """Returns the value array indices for the requested entity IDs
        
        Parameters
        ----------
        eids: Iterable[int]
            a Iterable of entity IDs to return the indices for
        """
        return np.array([self.eid_to_idx[i] for i in eids], dtype=np.uint32)

    def __getitem__(self, eid: Union[int,Iterable[int]]) -> npt.NDArray:
        if isinstance(eid,int):
            return self.get_values(self.eid_to_idx[eid])
        else:
            return self.get_values(self.map_eids(eid))

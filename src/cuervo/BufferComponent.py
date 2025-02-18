from typing import Union, Sequence, Optional, Literal, Annotated

import numpy as np
import numpy.typing as npt

from .Component import Component

class BufferComponent(Component):
    """A buffered component class for efficiently reusing memory for components
    on entities which are frequently created or destroyed.
    """
    shape: Sequence[int]=()
    dtype: np.dtype = np.float32

    def __class_getitem__(cls, length: int) -> type:
        cls.buflen = length
        return Annotated[cls, Literal[length]]
    
    def __init__(self):
        super().__init__()
        self.values = np.empty((self.buflen,*self.shape), dtype=self.dtype)
        self.active = np.zeros((self.buflen,), dtype=bool)
    
    def eidx_mask(self, eidx: Sequence[int]) -> npt.NDArray[bool]:
        mask = np.zeros_like(self.active, dtype=bool)
        mask[eidx] = True
        return mask & self.active
    
    def get_values(self, eidx: Optional[Sequence[int]] = None) -> npt.NDArray:
        if eidx is None: return self.values[self.active]
        return self.values[self.eidx_mask(eidx)]
    
    def set_values(self, values: npt.NDArray, eidx: Optional[Sequence[int]] = None):
        if eidx is None:
            self.values[self.active] = values
        else:
            self.values[self.eidx_mask(eidx)] = values

    def add_entity(
        self,
        eid: Union[int, Sequence[int]],
        value: npt.ArrayLike,
        batch: bool = False
    ):
        """Adds new entities to this component and sets the initial values.

        Parameters
        ----------
        eid: Union[int, Sequence[int]]
            the entity ID or sequence of IDs to add
        value: npt.ArrayLike
            the values to add for the provided entity/entities
        batch: bool, default=False
            if this flag is set, the eid is assumed to be a sequence of values
            and the first dimension of the `value` parameter is assumed to be
            the number of entities being added
            
        """

        # TODO: Handle this case. Allow for a fixed or growable buffer on init
        #       and either throw or grow here
        valid_idx = np.arange(len(self.values))[~self.active]
        if len(valid_idx) < 1:
            self.logger.error("Have to grow buffer!")
            exit()

        if batch:
            # FIXME: see the above TODO
            if len(valid_idx) < len(eid):
                self.logger.error("Have to grow buffer!")
                exit()

            self.entities.update(eid)
            eidx = valid_idx[:len(value)]
            self.values[eidx] = value
            self.eid_to_idx |= {e:idx for e,idx in zip(eid,eidx, strict=True)}
        else:
            self.entities.add(eid)
            eidx = valid_idx[0]
            self.values[eidx] = value
            self.eid_to_idx[eid] = eidx

        self.active[eidx] = True
    
    def remove_entity(self, eid: int):
        """Removes an entity and its value from the component
        
        Parameters
        ----------
        eid: int
            the entity ID to remove
        """
        self.entities.remove(eid)
        idx = self.eid_to_idx.pop(eid)
        self.active[idx] = False

"""Animal packs: what the model needs to know about one kind of animal (new file).

The model (scope_model.py) is the same for every animal. What changes from one animal to another is
declared here: the keypoint names, how many animals are in the arena, the torso used for body
position, the skeleton drawn on screen, the behaviors scored and the frame rate of the training
data. The model file of each animal (models/etholive/<animal>.pt) carries its pack, so a file always
says which animal it is for.

Mouse is the first pack; it is exactly the CalMS21 setup the hackathon build was trained on.
A new animal (for example flies) is one more Species entry plus its trained model file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import calms21
from . import evaluate
from . import skeleton


@dataclass(frozen=True)
class Species:
    name: str                  # short id, used in file names: models/etholive/<name>.pt
    label: str                 # shown to the user
    keypoints: tuple           # keypoint names, in the order the pose model returns them
    animals: tuple             # animal ids in the arena, in pose-model order
    roles: dict                # animal id -> role shown to the user
    torso_keypoints: tuple     # the keypoints whose mean is the body position
    skeleton_edges: tuple      # pairs of keypoint names drawn as lines
    behaviors: tuple           # scored behaviors; the model also predicts "other"
    fps: float                 # frame rate of the training data
    data: str                  # where the training labels come from

    @property
    def classes(self):
        return self.behaviors + ("other",)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: tuple(tuple(e) if isinstance(e, list) else e for e in v) if isinstance(v, list) else v
                      for k, v in d.items()})


MOUSE = Species(
    name="mouse",
    label="Mouse (2 mice, resident and intruder)",
    keypoints=calms21.KEYPOINTS,
    animals=calms21.MICE,
    roles=dict(calms21.MOUSE_ROLE),
    torso_keypoints=calms21.TORSO_KEYPOINTS,
    skeleton_edges=skeleton.PROVISIONAL_EDGES,
    behaviors=evaluate.BEHAVIORS,
    fps=calms21.FPS,
    data="CalMS21 Task 1 (Sun et al., 2021)",
)

def _fly():
    from . import flyvsfly as fv
    return Species(
        name="fly",
        label="Fruit fly (2 males, aggression)",
        keypoints=fv.KEYPOINTS,
        animals=fv.FLIES,
        roles={"a": "fly 1", "b": "fly 2"},
        torso_keypoints=fv.TORSO_KEYPOINTS,
        skeleton_edges=fv.SKELETON,
        behaviors=fv.BEHAVIORS,
        fps=fv.FPS,
        data="Caltech Fly vs. Fly, Aggression set (Eyjolfsdottir et al., 2014; CC0)",
    )


def _fly_courtship():
    """The same flies and points, another question: a male and a female, courtship behaviors. A separate model, so the fight model stays as it is."""
    from dataclasses import replace
    return replace(FLY, name="fly_courtship", label="Fruit fly (male and female, courtship)", roles={"a": "fly 1", "b": "fly 2"},
                   behaviors=("wing_extension", "circle", "copul_attempt", "copulation"),
                   data="Caltech Fly vs. Fly, Courtship set (Eyjolfsdottir et al., 2014; CC0)")


FLY = _fly()
FLY_COURTSHIP = _fly_courtship()
SPECIES = {s.name: s for s in (MOUSE, FLY, FLY_COURTSHIP)}
DEFAULT = MOUSE.name


def get(name=DEFAULT):
    try:
        return SPECIES[name]
    except KeyError:
        raise ValueError(f"no animal pack called {name!r}; known: {', '.join(SPECIES)}") from None
